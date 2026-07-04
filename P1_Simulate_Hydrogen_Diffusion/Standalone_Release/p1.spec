# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


APP_NAME = "WeldCraft - Simulate Hydrogen Diffusion"
PORTABLE_NAME = f"{APP_NAME}-portable"
ONEFILE_BUILD_NAME = f"{APP_NAME}-onefile-build"

STANDALONE_DIR = Path(SPEC).resolve().parent
P1_DIR = STANDALONE_DIR.parent
REPO_ROOT = P1_DIR.parent
ENTRY_SCRIPT = P1_DIR / "simulate_hydrogen_diffusion.py"
ICON_PATH = REPO_ROOT / "Resources" / "Images" / "WeldCraft.ico"
DEFAULT_SETTINGS_PATH = STANDALONE_DIR / "staging" / "settings" / "default_settings.json"
BAM_LOGO_PATH = REPO_ROOT / "Resources" / "Images" / "BAM Logo.png"

datas = [
    (str(DEFAULT_SETTINGS_PATH), "settings"),
    (str(ICON_PATH), "Resources/Images"),
    (str(BAM_LOGO_PATH), "Resources/Images"),
]
datas += collect_data_files("matplotlib")
datas += collect_data_files("h5py")

hiddenimports = ["sip"]
hiddenimports += collect_submodules("matplotlib.backends")
hiddenimports += collect_submodules("h5py")
hiddenimports += collect_submodules("numba")

a = Analysis(
    [str(ENTRY_SCRIPT)],
    pathex=[str(P1_DIR), str(REPO_ROOT / "Resources")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

onefile_exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=ONEFILE_BUILD_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(ICON_PATH)],
)

portable_exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(ICON_PATH)],
)

coll = COLLECT(
    portable_exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=PORTABLE_NAME,
)
