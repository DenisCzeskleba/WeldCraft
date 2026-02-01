import argparse
import subprocess
from pathlib import Path

def is_venv_folder(p: Path) -> bool:
    return (p / "pyvenv.cfg").is_file() and (p / "Scripts" / "python.exe").is_file()

def find_venvs(root: Path, max_depth: int):
    root = root.resolve()
    results = []

    # Fast-ish: search for pyvenv.cfg and then verify python.exe exists
    # Depth limiting: we skip paths that are too deep
    for cfg in root.rglob("pyvenv.cfg"):
        try:
            rel_parts = cfg.relative_to(root).parts
        except ValueError:
            continue
        if len(rel_parts) > max_depth:
            continue

        env_dir = cfg.parent
        if is_venv_folder(env_dir):
            results.append(env_dir)

    # De-dup / stable order
    results = sorted(set(results), key=lambda p: str(p).lower())
    return results

def freeze_env(env_dir: Path, out_dir: Path):
    py = env_dir / "Scripts" / "python.exe"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Name file based on env folder name + a short path hash to avoid collisions
    safe_name = env_dir.name.replace(" ", "_")
    out_file = out_dir / f"requirements_{safe_name}.txt"

    print(f"  -> freezing: {env_dir}")
    try:
        # Using python -m pip ensures we use pip from that env
        res = subprocess.run(
            [str(py), "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print("     !! timeout during pip freeze")
        return False
    except Exception as e:
        print(f"     !! failed to run pip freeze: {type(e).__name__}: {e}")
        return False

    if res.returncode != 0:
        print("     !! pip freeze returned error")
        err = (res.stderr or "").strip()
        if err:
            print("     stderr:", err.splitlines()[-1])
        return False

    out_file.write_text(res.stdout, encoding="utf-8")
    print(f"     saved: {out_file}")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help=r"Root folder to scan, e.g. F:\ or F:\01_BAM Simulations")
    ap.add_argument("--max-depth", type=int, default=8, help="Max relative depth to consider (default: 8)")
    ap.add_argument("--freeze", action="store_true", help="Run pip freeze for each found venv")
    ap.add_argument("--out", default="requirements_exports", help="Output folder for requirements files")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    venvs = find_venvs(root, args.max_depth)
    print(f"Found {len(venvs)} venv(s) under {root} (max depth {args.max_depth}):\n")

    for v in venvs:
        py = v / "Scripts" / "python.exe"
        print(f"- {v}")
        print(f"  python: {py}")

    if args.freeze and venvs:
        out_dir = Path(args.out)
        print(f"\nExporting requirements to: {out_dir.resolve()}\n")
        ok = 0
        for v in venvs:
            if freeze_env(v, out_dir):
                ok += 1
        print(f"\nDone. Exported {ok}/{len(venvs)} requirements files.")

if __name__ == "__main__":
    main()
