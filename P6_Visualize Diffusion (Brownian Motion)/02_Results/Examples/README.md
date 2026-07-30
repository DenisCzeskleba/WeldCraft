# P6 Published Examples

This directory contains the outward-facing example diagrams for P6 and the
compact HDF5 source used to regenerate them.

## Canonical source

`published_examples_source.h5` is derived from a completed 70-billion-step
source/sink simulation with multiple regions, a trap layer, and a circular
high-affinity spot. It preserves the complete simulation configuration, the
initial and final matrix snapshots, the final restart checkpoint, and all 7,001
recorded transport intervals. Machine-specific paths have been removed from
the published metadata.

The original simulation contained 7,001 saved matrices and was approximately
5.9 GB. The published source intentionally retains only:

- saved frame 0 at simulation step 0;
- saved frame 1 at simulation step 70,000,000,000.

Consequently, its spatial diagrams and final concentration profile are based
on the real final state. Its full time-resolved flux curve remains exact because
the much smaller scalar transport history is preserved separately.

## Difference from a normal simulation HDF5

A normal P6 result aligns every root `saved_steps` entry, matrix snapshot, and
transport interval one-to-one. This publication-optimized derivative is
intentionally different:

- `/snapshots` and `/saved_steps` contain the two retained matrix states;
- `/transport/saved_steps` contains all 7,001 original transport times;
- the interval datasets inside `/transport` align with
  `/transport/saved_steps`, not with `/snapshots`.

The schema is identified in the HDF5 attributes and supported by the current
P6 transport and diagram loaders. Software that assumes every HDF5 dataset has
the same leading dimension must account for this distinction.

The compact file cannot recreate the original animation or intermediate
concentration profiles. Future diagram styles that require matrix history need
either the original result or a different sparse derivative retaining more
matrix snapshots.

## Regenerating every published diagram

In `03_CodeBase/c3_Brown_Make_Diagram.py`, select:

```python
DIAGRAM_PRESET = "all_presets"
```

Then run:

```powershell
& "F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe" `
  ".\P6_Visualize Diffusion (Brownian Motion)\03_CodeBase\c3_Brown_Make_Diagram.py"
```

The batch reads the final snapshot from `published_examples_source.h5`,
discovers every ordinary diagram preset, and recreates the numbered PNG files
in this directory. `published_examples_manifest.json` records which PNGs are
managed by the batch. After a completely successful render, PNGs listed by the
previous manifest but no longer generated are deleted.

The ordering in
`01_Resources/Diagram_Presets/all_presets.py::BATCH_PRESET_ORDER` defines the
stable public numbers. Add new presets at the end; do not reorder existing
entries unless their published filenames are intentionally being changed.

## Public and local files

Every file directly inside this directory is visible to Git. Every subfolder
and all content inside it are ignored, so personal experiments and thesis
outputs should always be placed in subfolders.
