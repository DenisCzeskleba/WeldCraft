# Lattice Visualizer Standalone Release Builder

This directory contains the canonical local-only builder for standalone Lattice
Visualizer release assets.

## Build command

Run from any directory and always provide the release version:

```powershell
& "F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe" `
  "F:\100_WebSite and Hosted Projects\WeldCraft\P5_Lattice_Visualizer\Standalone_Release\build_standalone_release.py" `
  --version 0.3
```

The builder validates the interpreter and dependency versions, tests the source
application, creates a PyInstaller one-folder executable, tests configuration
discovery in the packaged application, creates both RAR files, and verifies
their contents.

For version `0.3`, the final local output is:

```text
dist/v0.3/
|-- Plain Files/
|-- Lattice-Visualizer-v0.3-Script.rar
`-- Lattice-Visualizer-v0.3-Portable.rar
```

Copy the contents of `Plain Files` into the standalone repository's existing
`Plain Files` directory when the user explicitly chooses to update it. Upload
the two RAR files when the user explicitly chooses to publish the GitHub
release. GitHub creates its own source ZIP and TAR.GZ from the release tag.

Building assets does not authorize committing, tagging, pushing, uploading,
publishing a GitHub release, or creating a Zenodo record.
