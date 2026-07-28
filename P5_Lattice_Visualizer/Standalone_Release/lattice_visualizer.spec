# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all


APP_NAME = "visualize_lattice"

STANDALONE_DIR = Path(SPEC).resolve().parent
P5_DIR = STANDALONE_DIR.parent
ENTRY_SCRIPT = P5_DIR / "visualize_lattice.py"

pyvista_datas, pyvista_binaries, pyvista_hiddenimports = collect_all("pyvista")
vtk_datas, vtk_binaries, vtk_hiddenimports = collect_all("vtkmodules")
yaml_datas, yaml_binaries, yaml_hiddenimports = collect_all("yaml")

a = Analysis(
    [str(ENTRY_SCRIPT)],
    pathex=[str(P5_DIR)],
    binaries=pyvista_binaries + vtk_binaries + yaml_binaries,
    datas=pyvista_datas + vtk_datas + yaml_datas,
    hiddenimports=pyvista_hiddenimports + vtk_hiddenimports + yaml_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PyQt5", "PySide2", "PySide6"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
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
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)
