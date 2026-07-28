"""Build and validate the local standalone Lattice Visualizer release assets."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


EXPECTED_INTERPRETER = Path(
    r"F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe"
)
REQUIRED_PACKAGES = {
    "numpy": "2.0.2",
    "pyvista": "0.46.5",
    "PyYAML": "6.0.3",
    "pyinstaller": "6.21.0",
}
STANDALONE_REQUIREMENTS = (
    "numpy==2.0.2\n"
    "pyvista==0.46.5\n"
    "PyYAML==6.0.3\n"
)
VERSION_PATTERN = re.compile(r"^v?(\d+\.\d+(?:\.\d+)?)$")

STANDALONE_DIR = Path(__file__).resolve().parent
P5_DIR = STANDALONE_DIR.parent
REPO_ROOT = P5_DIR.parent
SOURCE_PATH = P5_DIR / "visualize_lattice.py"
CONFIG_PATH = P5_DIR / "User Input.yaml"
README_PATH = P5_DIR / "ReadMe.txt"
REQUIREMENTS_PATH = P5_DIR / "requirements.txt"
LICENSE_PATH = REPO_ROOT / "LICENSE"
SPEC_PATH = STANDALONE_DIR / "lattice_visualizer.spec"
BUILD_DIR = STANDALONE_DIR / "build"
STAGING_DIR = STANDALONE_DIR / "staging"
DIST_DIR = STANDALONE_DIR / "dist"

PLAIN_FILE_NAMES = {
    "ReadMe.txt",
    "User Input.yaml",
    "visualize_lattice.py",
    "requirements.txt",
    "LICENSE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build local standalone Lattice Visualizer release assets."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Release version such as 0.3 or 0.3.0. A leading v is optional.",
    )
    return parser.parse_args()


def normalize_version(value: str) -> str:
    match = VERSION_PATTERN.fullmatch(value.strip())
    if not match:
        raise ValueError(
            "Version must contain two or three numeric components, "
            "for example 0.3 or 0.3.0."
        )
    return f"v{match.group(1)}"


def normalized_path(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def require_path_within(path: Path, parent: Path) -> None:
    resolved_path = normalized_path(path)
    resolved_parent = normalized_path(parent)
    if resolved_path == resolved_parent:
        raise ValueError(f"Refusing to treat the allowed root as an output: {path}")
    if not resolved_path.startswith(resolved_parent + os.sep):
        raise ValueError(f"Output path resolved outside {parent}: {path}")


def reset_directory(path: Path, allowed_parent: Path) -> None:
    require_path_within(path, allowed_parent)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def validate_interpreter() -> None:
    if normalized_path(Path(sys.executable)) != normalized_path(EXPECTED_INTERPRETER):
        raise RuntimeError(
            "Run this builder with the WeldCraft interpreter:\n"
            f"{EXPECTED_INTERPRETER}\n"
            f"Current interpreter: {sys.executable}"
        )


def validate_packages() -> None:
    mismatches = []
    for package, expected_version in REQUIRED_PACKAGES.items():
        try:
            actual_version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{package}: missing (expected {expected_version})")
            continue
        if actual_version != expected_version:
            mismatches.append(
                f"{package}: {actual_version} installed (expected {expected_version})"
            )
    if mismatches:
        raise RuntimeError(
            "Standalone build dependencies are not reproducible:\n"
            + "\n".join(mismatches)
        )


def find_rar_executable() -> Path:
    candidates = [
        Path(r"C:\Program Files\WinRAR\Rar.exe"),
        Path(r"C:\Program Files\WinRAR\WinRAR.exe"),
    ]
    path_from_environment = shutil.which("rar")
    if path_from_environment:
        candidates.insert(0, Path(path_from_environment))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "WinRAR command-line utility not found. Expected Rar.exe in PATH or "
        r"C:\Program Files\WinRAR."
    )


def validate_inputs() -> None:
    required_paths = [
        SOURCE_PATH,
        CONFIG_PATH,
        README_PATH,
        REQUIREMENTS_PATH,
        LICENSE_PATH,
        SPEC_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required release inputs are missing:\n" + "\n".join(missing))

    ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{CONFIG_PATH} must contain a YAML mapping.")

    requirements = REQUIREMENTS_PATH.read_text(encoding="utf-8")
    if requirements != STANDALONE_REQUIREMENTS:
        raise ValueError(
            f"{REQUIREMENTS_PATH} does not match the approved standalone dependency pins."
        )

    license_text = LICENSE_PATH.read_text(encoding="utf-8")
    if "MIT License" not in license_text or "Permission is hereby granted" not in license_text:
        raise ValueError(f"{LICENSE_PATH} is not the expected MIT license.")


def run_checked(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def validate_source_import() -> None:
    code = (
        "import runpy; "
        f"runpy.run_path({str(SOURCE_PATH)!r}, run_name='lattice_visualizer_import_check')"
    )
    run_checked([sys.executable, "-c", code], cwd=P5_DIR)


def create_small_test_config(path: Path) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["target_atoms"] = 1
    config["demo_cell_force"] = True
    config["Nx"] = 1
    config["Ny"] = 1
    config["Nz"] = 1
    path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )


def smoke_test_source(small_config_path: Path) -> None:
    run_checked(
        [
            sys.executable,
            str(SOURCE_PATH),
            "--config",
            str(small_config_path),
            "--no-show",
        ],
        cwd=STAGING_DIR,
    )


def test_source_config_discovery() -> None:
    unrelated_directory = STAGING_DIR / "unrelated-working-directory"
    unrelated_directory.mkdir(parents=True, exist_ok=True)
    code = "\n".join(
        [
            "import importlib.util",
            "from pathlib import Path",
            f"source = Path({str(SOURCE_PATH)!r})",
            "spec = importlib.util.spec_from_file_location('lattice_config_check', source)",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "found = Path(module.guess_default_config()).resolve()",
            f"expected = Path({str(CONFIG_PATH)!r}).resolve()",
            "assert found == expected, f'expected {expected}, found {found}'",
        ]
    )
    run_checked([sys.executable, "-c", code], cwd=unrelated_directory)


def prepare_plain_files(plain_files_dir: Path) -> None:
    plain_files_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(README_PATH, plain_files_dir / "ReadMe.txt")
    shutil.copy2(CONFIG_PATH, plain_files_dir / "User Input.yaml")
    shutil.copy2(SOURCE_PATH, plain_files_dir / "visualize_lattice.py")
    shutil.copy2(REQUIREMENTS_PATH, plain_files_dir / "requirements.txt")
    shutil.copy2(LICENSE_PATH, plain_files_dir / "LICENSE")


def run_pyinstaller() -> Path:
    pyinstaller_dist = STAGING_DIR / "pyinstaller-dist"
    pyinstaller_dist.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            str(pyinstaller_dist),
            "--workpath",
            str(BUILD_DIR),
            str(SPEC_PATH),
        ],
        cwd=STANDALONE_DIR,
    )

    portable_dir = pyinstaller_dist / "visualize_lattice"
    expected_exe = portable_dir / "visualize_lattice.exe"
    internal_dir = portable_dir / "_internal"
    if not expected_exe.is_file() or not internal_dir.is_dir():
        raise FileNotFoundError(
            "PyInstaller did not create the expected one-folder distribution."
        )
    return portable_dir


def add_portable_support_files(portable_dir: Path) -> None:
    shutil.copy2(README_PATH, portable_dir / "ReadMe.txt")
    shutil.copy2(CONFIG_PATH, portable_dir / "User Input.yaml")
    shutil.copy2(REQUIREMENTS_PATH, portable_dir / "requirements.txt")
    shutil.copy2(LICENSE_PATH, portable_dir / "LICENSE")


def smoke_test_portable(portable_dir: Path, small_config_path: Path) -> None:
    portable_config = portable_dir / "User Input.yaml"
    original_config = portable_config.read_bytes()
    unrelated_directory = STAGING_DIR / "portable-unrelated-working-directory"
    unrelated_directory.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(small_config_path, portable_config)
        run_checked(
            [str(portable_dir / "visualize_lattice.exe"), "--no-show"],
            cwd=unrelated_directory,
        )
    finally:
        portable_config.write_bytes(original_config)


def create_rar(
    rar_executable: Path,
    archive_path: Path,
    *,
    working_directory: Path,
    source_name: str,
) -> None:
    run_checked(
        [
            str(rar_executable),
            "a",
            "-ma5",
            "-idq",
            str(archive_path),
            source_name,
        ],
        cwd=working_directory,
    )
    if not archive_path.is_file():
        raise FileNotFoundError(f"WinRAR did not create {archive_path}")


def list_rar_entries(rar_executable: Path, archive_path: Path) -> set[str]:
    result = subprocess.run(
        [str(rar_executable), "lb", str(archive_path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        line.strip().replace("/", "\\")
        for line in result.stdout.splitlines()
        if line.strip()
    }


def validate_script_archive(rar_executable: Path, archive_path: Path) -> None:
    entries = list_rar_entries(rar_executable, archive_path)
    if entries != PLAIN_FILE_NAMES:
        raise RuntimeError(
            "Script archive contents differ from the release contract.\n"
            f"Expected: {sorted(PLAIN_FILE_NAMES)}\n"
            f"Actual: {sorted(entries)}"
        )


def validate_portable_archive(rar_executable: Path, archive_path: Path) -> None:
    entries = list_rar_entries(rar_executable, archive_path)
    prefix = "visualize_lattice\\"
    required = {
        prefix + "ReadMe.txt",
        prefix + "User Input.yaml",
        prefix + "requirements.txt",
        prefix + "LICENSE",
        prefix + "visualize_lattice.exe",
    }
    missing = required - entries
    unexpected = {
        entry
        for entry in entries
        if entry != "visualize_lattice"
        and entry != prefix + "_internal"
        and not entry.startswith(prefix + "_internal\\")
        and entry not in required
    }
    internal_files = {
        entry for entry in entries if entry.startswith(prefix + "_internal\\")
    }
    if missing or unexpected or not internal_files:
        raise RuntimeError(
            "Portable archive contents differ from the release contract.\n"
            f"Missing: {sorted(missing)}\n"
            f"Unexpected: {sorted(unexpected)}\n"
            f"Internal files found: {len(internal_files)}"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_size(path: Path) -> str:
    return f"{path.stat().st_size / (1024 * 1024):.2f} MiB"


def main() -> None:
    args = parse_args()
    version = normalize_version(args.version)
    output_dir = DIST_DIR / version

    validate_interpreter()
    validate_packages()
    validate_inputs()
    rar_executable = find_rar_executable()

    STANDALONE_DIR.mkdir(parents=True, exist_ok=True)
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    reset_directory(BUILD_DIR, STANDALONE_DIR)
    reset_directory(STAGING_DIR, STANDALONE_DIR)
    reset_directory(output_dir, DIST_DIR)

    print(f"Building Lattice Visualizer standalone assets for {version}...")

    small_config_path = STAGING_DIR / "small-test-config.yaml"
    create_small_test_config(small_config_path)
    validate_source_import()
    smoke_test_source(small_config_path)
    test_source_config_discovery()

    plain_files_dir = output_dir / "Plain Files"
    prepare_plain_files(plain_files_dir)

    portable_dir = run_pyinstaller()
    add_portable_support_files(portable_dir)
    smoke_test_portable(portable_dir, small_config_path)

    script_archive = output_dir / f"Lattice-Visualizer-{version}-Script.rar"
    portable_archive = output_dir / f"Lattice-Visualizer-{version}-Portable.rar"
    create_rar(
        rar_executable,
        script_archive,
        working_directory=plain_files_dir,
        source_name="*",
    )
    create_rar(
        rar_executable,
        portable_archive,
        working_directory=portable_dir.parent,
        source_name=portable_dir.name,
    )

    validate_script_archive(rar_executable, script_archive)
    validate_portable_archive(rar_executable, portable_archive)

    print("\nStandalone release assets are ready for manual publication.")
    print(f"Plain files: {plain_files_dir}")
    for archive in (script_archive, portable_archive):
        print(f"{archive.name}: {archive}")
        print(f"  Size: {format_size(archive)}")
        print(f"  SHA-256: {sha256(archive)}")
    print("\nNo repository, tag, GitHub release, upload, or Zenodo record was changed.")
    print("GitHub will generate Source code (zip) and Source code (tar.gz) from the release tag.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
