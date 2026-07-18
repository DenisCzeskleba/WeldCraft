# P6 Brownian Motion User Guide

This resource README describes the simulation choices available to P6 users. It is separate from the
WeldCraft repository README.

P6 provides three movement modes. They share the same matrix initialization, concentration settings,
spot/layer topology, HDF5 snapshot format, and plotting tools. They differ in how molecular movement is
executed.

## `molecular_wiggle`

This is the direct, synchronous wiggle model. During one step, every hydrogen atom proposes one random
X/Y displacement. The displacement is accepted according to the Gaussian distance probability. All
proposals inspect the state at the beginning of the step; when several atoms request one empty site, a
random priority selects the winner. Accepted winners are applied together.

One step means one proposal opportunity for every hydrogen atom present at the beginning of the step.

## `random_sequential_wiggle`

This is an optimized form of `molecular_wiggle`, not a different movement probability. During startup,
P6 precomputes the marginal probability of every reachable source-site to destination-site wiggle. All
random coordinates that would miss a site, leave the matrix, or fail the Gaussian test are represented by
the remaining probability of staying in place.

The simulation stores only compact site states and a dynamic list of hydrogen site IDs. It reconstructs
the full matrix only when a snapshot is saved. Hydrogen atoms are selected with replacement and moves
are applied immediately, so simultaneous collisions and execution-order conflict rules are unnecessary.

For this mode, one step contains as many random hydrogen selections as there are hydrogen atoms at the
start of that step. Consequently, every atom receives one wiggle opportunity per step on average. The
existing `steps` and `save_every_steps` settings retain this meaning, so `save_every_steps = 25000` saves
at intervals of 25,000 average wiggle opportunities per atom.

The HDF5 file also stores `wiggle_attempt_count` and `hydrogen_count` for every snapshot. These preserve
the exact number of individual selections and the changing population behind the convenient step label.

When sink/source boundaries are active:

- A normal move updates the selected hydrogen's site ID.
- A hydrogen leaving the source keeps its source site occupied and appends a new hydrogen at the destination.
- A hydrogen entering the sink empties its origin, leaves the sink empty, and is swap-removed from the selectable list.

## `forced_jump`

This mode selects from currently valid available destination sites within the configured radius. If a
valid destination exists, the hydrogen jumps without the Gaussian proposal/rejection behavior used by
the two wiggle modes. Its compact precomputed lane is used when sink/source boundaries are disabled; a
matrix-scanning safe lane is used when those boundaries can change the hydrogen population.

## Reproducibility

All modes record the actual random seed and algorithm in HDF5 metadata. Set `random_seed` to an integer
to reproduce initialization and movement. Leave it as `None` to create and record a fresh seed per run.
