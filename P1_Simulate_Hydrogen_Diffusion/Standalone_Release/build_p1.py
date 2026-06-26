import shutil
import subprocess
import sys
from pathlib import Path


APP_NAME = "WeldCraft - Simulate Hydrogen Diffusion"
PORTABLE_NAME = f"{APP_NAME}-portable"
ONEFILE_BUILD_NAME = f"{APP_NAME}-onefile-build"

STANDALONE_DIR = Path(__file__).resolve().parent
P1_DIR = STANDALONE_DIR.parent
APP_FILES_DIR = P1_DIR / "App_Files"
SETTINGS_JSON_PATH = APP_FILES_DIR / "settings.json"
SPEC_PATH = STANDALONE_DIR / "p1.spec"
BUILD_DIR = STANDALONE_DIR / "build"
DIST_DIR = STANDALONE_DIR / "dist"
STAGING_DIR = STANDALONE_DIR / "staging"


def prepare_staging_assets():
    staging_settings_dir = STAGING_DIR / "settings"
    staging_settings_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SETTINGS_JSON_PATH, staging_settings_dir / "default_settings.json")


def reset_output_directories():
    for path in (BUILD_DIR, DIST_DIR, STAGING_DIR):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def run_pyinstaller():
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR),
        str(SPEC_PATH),
    ]
    subprocess.run(command, cwd=STANDALONE_DIR, check=True)


def finalize_onefile_name():
    temporary_exe = DIST_DIR / f"{ONEFILE_BUILD_NAME}.exe"
    final_exe = DIST_DIR / f"{APP_NAME}.exe"
    if final_exe.exists():
        final_exe.unlink()
    temporary_exe.replace(final_exe)


def seed_portable_runtime_layout():
    portable_dir = DIST_DIR / PORTABLE_NAME
    settings_dir = portable_dir / "settings"
    results_dir = portable_dir / "Results"
    settings_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SETTINGS_JSON_PATH, settings_dir / "settings.json")


def zip_portable_bundle():
    archive_base = DIST_DIR / PORTABLE_NAME
    if archive_base.with_suffix(".zip").exists():
        archive_base.with_suffix(".zip").unlink()
    shutil.make_archive(
        str(archive_base),
        "zip",
        root_dir=DIST_DIR,
        base_dir=PORTABLE_NAME,
    )


def validate_outputs():
    onefile_exe = DIST_DIR / f"{APP_NAME}.exe"
    portable_exe = DIST_DIR / PORTABLE_NAME / f"{APP_NAME}.exe"
    portable_zip = DIST_DIR / f"{PORTABLE_NAME}.zip"
    expected_paths = [onefile_exe, portable_exe, portable_zip]
    missing_paths = [str(path) for path in expected_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Standalone build completed but expected outputs are missing:\n"
            + "\n".join(missing_paths)
        )


def main():
    print(f"Building {APP_NAME} standalone release assets...")
    reset_output_directories()
    prepare_staging_assets()
    run_pyinstaller()
    finalize_onefile_name()
    seed_portable_runtime_layout()
    zip_portable_bundle()
    validate_outputs()
    print("Build completed.")
    print(f"Onefile EXE: {DIST_DIR / f'{APP_NAME}.exe'}")
    print(f"Portable folder: {DIST_DIR / PORTABLE_NAME}")
    print(f"Portable ZIP: {DIST_DIR / f'{PORTABLE_NAME}.zip'}")


if __name__ == "__main__":
    main()
