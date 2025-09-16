# Batch_Run.py
import os
import sys
import shutil
import subprocess
import signal
import atexit
from typing import Tuple

CONFIG_FILE = "param_config.py"
BACKUP_FILE = "param_config_backup.py"
BATCH_DIR   = "Batch executions"
STOP_ON_ERROR = True   # stop the sweep on first failed run

# ---- internal state ----
_RESTORED = False  # guard so we only restore once

# ---------- helpers ----------
def backup_config():
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"{CONFIG_FILE} not found.")
    shutil.copy2(CONFIG_FILE, BACKUP_FILE)

def restore_config_if_needed():
    """Idempotent restore: only runs once even if called multiple times."""
    global _RESTORED
    if _RESTORED:
        return
    if os.path.exists(BACKUP_FILE):
        try:
            shutil.move(BACKUP_FILE, CONFIG_FILE)
            print("[INFO] param_config.py restored from backup.")
        except Exception as e:
            print(f"[WARN] Failed to restore {CONFIG_FILE}: {e}")
    else:
        print("[WARN] Backup not found; param_config.py left as-is.")
    _RESTORED = True

# atexit: try to restore on any interpreter exit
atexit.register(restore_config_if_needed)

# signals: catch Ctrl-C / IDE stop, restore, and exit with code 130
def _signal_handler(signum, frame):
    print(f"\n[INFO] Caught signal {signum}. Restoring config and exiting...")
    restore_config_if_needed()
    # 130 is conventional for SIGINT exits
    sys.exit(130)

# Windows/POSIX coverage (SIGTERM may be a no-op on some Windows setups)
for sig in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
    try:
        signal.signal(sig, _signal_handler)
    except Exception:
        pass  # some environments don’t allow setting handlers

def sanitize_path_for_py(s: str) -> str:
    """Use forward slashes so the string is safe inside Python code on Windows."""
    return s.replace("\\", "/")

def value_token(v):
    """Short token for filenames from a value."""
    if isinstance(v, (int, float)):
        s = f"{v:g}"            # compact numeric (e.g., 1.5, 1e-3)
        s = s.replace(".", "p") # 1.5 -> 1p5
        s = s.replace("-", "m") # -3  -> m3 (avoid minus in filenames)
        return s
    # string: keep as-is but strip spaces
    return str(v).strip().replace(" ", "_")

def build_output_names(changes: dict) -> Tuple[str, str]:
    """Build output file names that encode the changed params (except file_* fields)."""
    parts = []
    for k in sorted(changes.keys()):
        if k in {"file_name", "animation_name"}:
            continue
        parts.append(f"{k}-{value_token(changes[k])}")
    suffix = "_".join(parts) if parts else "run"
    file_name = os.path.join(BATCH_DIR, f"diffusion_array_{suffix}.h5")
    anim_name = os.path.join(BATCH_DIR, f"animation_{suffix}.mp4")
    return sanitize_path_for_py(file_name), sanitize_path_for_py(anim_name)

def modify_config(changes: dict):
    """
    Update existing assignments in param_config.py.
    Keeps trailing comments. Supports numbers & strings.
    Only matches lines that start with `key =`.
    """
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    change_keys = set(changes.keys())

    for line in lines:
        stripped = line.lstrip()  # keep leading indentation if any
        updated = False
        for key in list(change_keys):
            # we only rewrite if the line starts with `key =`
            if stripped.startswith(f"{key} ="):
                # preserve indentation
                indent = line[:len(line) - len(stripped)]
                # preserve trailing comment if any
                comment = ""
                if "#" in line:
                    comment = "  #" + line.split("#", 1)[1].rstrip("\n")

                val = changes[key]
                if isinstance(val, str):
                    new_line = f'{indent}{key} = "{val}"{comment}\n'
                else:
                    new_line = f"{indent}{key} = {val}{comment}\n"

                new_lines.append(new_line)
                updated = True
                break
        if not updated:
            new_lines.append(line)

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

def run_script(script_name: str) -> int:
    try:
        subprocess.run([sys.executable, script_name], check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {script_name} failed: {e}")
        return e.returncode

# ---------- define your sweep here ----------
if __name__ == "__main__":
    os.makedirs(BATCH_DIR, exist_ok=True)

    # COMMON changes across all runs (optional)
    base_changes = {
        # "weld_simulation_style": "ellipse",   # example of string change
    }

    # PARAM SWEEP:
    h_inside_values = [0, 5, 10, 15, 20, 30, 40, 50, 60, 75, 80, 90, 100, 150, 200, 500, 1000]

    # Build runs you can extend later
    runs = []
    for h_val in h_inside_values:
        runs.append({
            "h_on_the_inside": h_val,
            # add more keys later if needed
        })

    # ---------- execute batch ----------
    backup_config()
    try:
        for i, run_changes in enumerate(runs, start=1):
            # Merge base changes + run changes
            changes = {**base_changes, **run_changes}

            # Build output names that encode the changed params
            file_name, animation_name = build_output_names(changes)
            changes["file_name"] = file_name
            changes["animation_name"] = animation_name

            # Apply changes to param_config.py
            modify_config(changes)

            print(f"\n=== Batch run {i}/{len(runs)} ===")
            print("param changes:", changes)

            rc1 = run_script("Heat_PDE_LDDE.py")
            if rc1 != 0 and STOP_ON_ERROR:
                print("[INFO] Stopping batch due to error in Heat_PDE_LDDE.py")
                break

            rc2 = run_script("Make_Animation.py")
            if rc2 != 0 and STOP_ON_ERROR:
                print("[INFO] Stopping batch due to error in Make_Animation.py")
                break

        print(f"\n[INFO] Batch finished (completed {i} run(s)). Results in: {BATCH_DIR}")
    finally:
        # final safety restore (also handled by atexit/handlers if we’re killed)
        restore_config_if_needed()
