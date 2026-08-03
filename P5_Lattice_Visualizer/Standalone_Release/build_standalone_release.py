"""Build and validate the local standalone Lattice Visualizer release assets."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import importlib.metadata
import os
import pprint
import re
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED_INTERPRETER = Path(
    r"F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe"
)
REQUIRED_PACKAGES = {
    "numpy": "2.0.2",
    "pyvista": "0.46.5",
    "PyQt5": "5.15.11",
    "PyQt5-Qt5": "5.15.2",
    "PyQt5_sip": "12.15.0",
    "pyinstaller": "6.21.0",
}
STANDALONE_REQUIREMENTS = (
    "numpy==2.0.2\n"
    "pyvista==0.46.5\n"
    "PyQt5==5.15.11\n"
    "PyQt5-Qt5==5.15.2\n"
    "PyQt5_sip==12.15.0\n"
)
VERSION_PATTERN = re.compile(r"^v?(\d+\.\d+(?:\.\d+)?)$")

STANDALONE_DIR = Path(__file__).resolve().parent
P5_DIR = STANDALONE_DIR.parent
REPO_ROOT = P5_DIR.parent
SOURCE_PATH = P5_DIR / "visualize_lattice.py"
GUI_SOURCE_PATH = P5_DIR / "lattice_visualizer_gui.py"
DEFAULT_CONFIG_PATH = P5_DIR / "01_Resources" / "config_default.py"
BAM_LOGO_PATH = P5_DIR / "01_Resources" / "Images" / "BAM Logo.png"
README_PATH = P5_DIR / "ReadMe.txt"
REQUIREMENTS_PATH = P5_DIR / "requirements.txt"
LICENSE_PATH = REPO_ROOT / "LICENSE"
SPEC_PATH = STANDALONE_DIR / "lattice_visualizer.spec"
RENDERER_SPEC_PATH = STANDALONE_DIR / "lattice_visualizer_renderer.spec"
BUILD_DIR = STANDALONE_DIR / "build"
STAGING_DIR = STANDALONE_DIR / "staging"
DIST_DIR = STANDALONE_DIR / "dist"

PLAIN_FILE_NAMES = {
    "ReadMe.txt",
    "config.py",
    "config_default.py",
    "lattice_visualizer_gui.py",
    "visualize_lattice.py",
    "requirements.txt",
    "LICENSE",
    "01_Resources\\config_default.py",
    "01_Resources\\Images\\BAM Logo.png",
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
        GUI_SOURCE_PATH,
        DEFAULT_CONFIG_PATH,
        BAM_LOGO_PATH,
        README_PATH,
        REQUIREMENTS_PATH,
        LICENSE_PATH,
        SPEC_PATH,
        RENDERER_SPEC_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required release inputs are missing:\n" + "\n".join(missing))

    ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))

    ast.parse(GUI_SOURCE_PATH.read_text(encoding="utf-8"), filename=str(GUI_SOURCE_PATH))
    default_tree = ast.parse(
        DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"),
        filename=str(DEFAULT_CONFIG_PATH),
    )
    settings_assignments = [
        node for node in default_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SETTINGS" for target in node.targets)
    ]
    if not settings_assignments:
        raise ValueError(f"{DEFAULT_CONFIG_PATH} must define SETTINGS.")

    requirements = REQUIREMENTS_PATH.read_text(encoding="utf-8")
    if requirements != STANDALONE_REQUIREMENTS:
        raise ValueError(
            f"{REQUIREMENTS_PATH} does not match the approved standalone dependency pins."
        )

    license_text = LICENSE_PATH.read_text(encoding="utf-8")
    if "MIT License" not in license_text or "Permission is hereby granted" not in license_text:
        raise ValueError(f"{LICENSE_PATH} is not the expected MIT license.")


def load_default_settings() -> dict:
    spec = importlib.util.spec_from_file_location("p5_release_config_default", DEFAULT_CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {DEFAULT_CONFIG_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = getattr(module, "SETTINGS", None)
    if not isinstance(settings, dict):
        raise ValueError(f"{DEFAULT_CONFIG_PATH} must define a SETTINGS dictionary")
    return copy.deepcopy(settings)


def write_config_module(path: Path, settings: dict) -> None:
    """Write a config using the documented template and preserve its comments."""

    template = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(template, filename=str(DEFAULT_CONFIG_PATH))
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SETTINGS" for target in node.targets)
    )
    if not isinstance(assignment.value, ast.Dict):
        raise ValueError("The default config SETTINGS must be a dictionary literal")

    lines = template.splitlines(keepends=True)
    starts = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)

    def position(line, column):
        return starts[line - 1] + column

    replacements = []
    for key_node, value_node in zip(assignment.value.keys, assignment.value.values):
        if not isinstance(key_node, ast.Constant) or key_node.value not in settings:
            continue
        start = position(value_node.lineno, value_node.col_offset)
        end = position(value_node.end_lineno, value_node.end_col_offset)
        replacements.append(
            (start, end, pprint.pformat(settings[key_node.value], sort_dicts=False, width=112))
        )
    for start, end, replacement in reversed(replacements):
        template = template[:start] + replacement + template[end:]
    path.write_text(template, encoding="utf-8", newline="\n")


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
    config = load_default_settings()
    config["target_atoms"] = 1
    config["demo_cell_force"] = True
    # Smoke tests validate startup/rendering without leaving user-output
    # directories inside the portable release contract.
    config["display_window"] = False
    config["save_png"] = False
    config["Nx"] = 1
    config["Ny"] = 1
    config["Nz"] = 1
    write_config_module(path, config)


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
    # A generic config.py in the caller's directory must not override P5's
    # persistent config beside the source script.
    (unrelated_directory / "config.py").write_text(
        "SETTINGS = {'lattice': 'unrelated'}\n",
        encoding="utf-8",
    )
    code = "\n".join(
        [
            "import importlib.util",
            "from pathlib import Path",
            f"source = Path({str(SOURCE_PATH)!r})",
            "spec = importlib.util.spec_from_file_location('lattice_config_check', source)",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "module.ensure_config_file()",
            "found = Path(module.guess_default_config()).resolve()",
            "expected = Path(module.persistent_config_path()).resolve()",
            "assert found == expected, f'expected {expected}, found {found}'",
        ]
    )
    run_checked([sys.executable, "-c", code], cwd=unrelated_directory)


def prepare_plain_files(plain_files_dir: Path) -> None:
    plain_files_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(README_PATH, plain_files_dir / "ReadMe.txt")
    shutil.copy2(DEFAULT_CONFIG_PATH, plain_files_dir / "config.py")
    shutil.copy2(DEFAULT_CONFIG_PATH, plain_files_dir / "config_default.py")
    resources_dir = plain_files_dir / "01_Resources"
    images_dir = resources_dir / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DEFAULT_CONFIG_PATH, resources_dir / "config_default.py")
    shutil.copy2(BAM_LOGO_PATH, images_dir / "BAM Logo.png")
    shutil.copy2(GUI_SOURCE_PATH, plain_files_dir / "lattice_visualizer_gui.py")
    shutil.copy2(SOURCE_PATH, plain_files_dir / "visualize_lattice.py")
    shutil.copy2(REQUIREMENTS_PATH, plain_files_dir / "requirements.txt")
    shutil.copy2(LICENSE_PATH, plain_files_dir / "LICENSE")


def run_pyinstaller() -> Path:
    gui_dist = STAGING_DIR / "gui-pyinstaller-dist"
    renderer_dist = STAGING_DIR / "renderer-pyinstaller-dist"
    gui_work = BUILD_DIR / "gui"
    renderer_work = BUILD_DIR / "renderer"
    for path in (gui_dist, renderer_dist, gui_work, renderer_work):
        path.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            str(gui_dist),
            "--workpath",
            str(gui_work),
            str(SPEC_PATH),
        ],
        cwd=STANDALONE_DIR,
    )

    run_checked(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            str(renderer_dist),
            "--workpath",
            str(renderer_work),
            str(RENDERER_SPEC_PATH),
        ],
        cwd=STANDALONE_DIR,
    )

    gui_dir = gui_dist / "visualize_lattice"
    renderer_dir = renderer_dist / "visualize_lattice_renderer"
    portable_dir = STAGING_DIR / "pyinstaller-dist" / "visualize_lattice"
    if portable_dir.exists():
        shutil.rmtree(portable_dir)
    shutil.copytree(gui_dir, portable_dir)
    expected_exe = portable_dir / "visualize_lattice.exe"
    renderer_exe = renderer_dir / "visualize_lattice_renderer.exe"
    internal_dir = portable_dir / "_internal"
    if not expected_exe.is_file() or not renderer_exe.is_file() or not internal_dir.is_dir():
        raise FileNotFoundError("PyInstaller did not create the coordinated GUI/renderer distribution.")
    shutil.copy2(renderer_exe, portable_dir / renderer_exe.name)
    return portable_dir


def add_portable_support_files(portable_dir: Path) -> None:
    shutil.copy2(README_PATH, portable_dir / "ReadMe.txt")
    shutil.copy2(DEFAULT_CONFIG_PATH, portable_dir / "config.py")
    resources_dir = portable_dir / "01_Resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DEFAULT_CONFIG_PATH, resources_dir / "config_default.py")
    images_dir = resources_dir / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BAM_LOGO_PATH, images_dir / "BAM Logo.png")
    shutil.copy2(REQUIREMENTS_PATH, portable_dir / "requirements.txt")
    shutil.copy2(LICENSE_PATH, portable_dir / "LICENSE")


def smoke_test_portable(portable_dir: Path, small_config_path: Path) -> None:
    portable_config = portable_dir / "config.py"
    original_config = portable_config.read_bytes()
    unrelated_directory = STAGING_DIR / "portable-unrelated-working-directory"
    unrelated_directory.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(small_config_path, portable_config)
        run_checked(
            [str(portable_dir / "visualize_lattice_renderer.exe"), "--no-show"],
            cwd=unrelated_directory,
        )
        # Exercise the GUI executable's own frozen-mode path resolution. This
        # verifies that it can locate and launch the sibling renderer, rather
        # than merely proving that the renderer works when invoked directly.
        run_checked(
            [str(portable_dir / "visualize_lattice.exe"), "--smoke-test-renderer"],
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
            "-r",
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
    # Recursive RAR creation includes explicit directory records. Validate the
    # files as the release contract and allow only the two expected resource
    # directories in addition to them.
    expected_directories = {"01_Resources", "01_Resources\\Images"}
    files = entries - expected_directories
    unexpected_directories = {
        entry for entry in entries
        if entry not in PLAIN_FILE_NAMES and entry not in expected_directories
    }
    if files != PLAIN_FILE_NAMES or unexpected_directories:
        raise RuntimeError(
            "Script archive contents differ from the release contract.\n"
            f"Expected: {sorted(PLAIN_FILE_NAMES)}\n"
            f"Actual files: {sorted(files)}\n"
            f"Unexpected directories: {sorted(unexpected_directories)}"
        )


def validate_portable_archive(rar_executable: Path, archive_path: Path) -> None:
    entries = list_rar_entries(rar_executable, archive_path)
    prefix = "visualize_lattice\\"
    required = {
        prefix + "ReadMe.txt",
        prefix + "config.py",
        prefix + "01_Resources\\config_default.py",
        prefix + "01_Resources\\Images\\BAM Logo.png",
        prefix + "requirements.txt",
        prefix + "LICENSE",
        prefix + "visualize_lattice.exe",
        prefix + "visualize_lattice_renderer.exe",
    }
    missing = required - entries
    unexpected = {
        entry
        for entry in entries
        if entry != "visualize_lattice"
        and entry != prefix + "_internal"
        and entry != prefix + "01_Resources"
        and entry != prefix + "01_Resources\\Images"
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

    small_config_path = STAGING_DIR / "small-test-config.py"
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
