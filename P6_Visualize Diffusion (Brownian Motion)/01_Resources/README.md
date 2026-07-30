# P6 Brownian Motion User Guide

This resource README describes the simulation choices available to P6 users. It is separate from the
WeldCraft repository README.

P6 provides three supported wiggle-derived movement modes plus the deprecated `forced_jump` legacy mode.
They share the same matrix initialization, concentration settings, spot/layer topology, HDF5 snapshot
format, and plotting tools. The supported modes differ in how molecular movement is executed.

## Saving and continuing a simulation

Every new run saves the segment's starting state, each configured interval, and the exact final state.
The endpoint is included once even when it already falls on `save_every_steps`. The final snapshot and
checkpoint therefore describe the same global step.

To continue a completed checkpoint-enabled file, keep its dynamics settings unchanged and configure:

```python
RESUME_FROM_H5 = r"F:\path\to\completed_run.h5"
h5_filename = "continued_run.h5"  # Must be different from the source file
steps = 15_000_000_000            # Additional steps, not the new absolute total
```

A relative `RESUME_FROM_H5` path is resolved under `02_Results`. The continuation is always written as
a new HDF5 segment and its `saved_steps` retain global step numbers. P6 validates the movement mode,
geometry-dependent area settings, source/sink settings, and movement characteristics before starting.

New completed files contain an exact checkpoint. It includes the aligned final matrix, RNG state, and
the internal ordering/state needed by the selected mode. Exact continuation therefore produces the same
state as an uninterrupted run, including when the first segment ends between normal save intervals.

Older files and files stopped with Ctrl+C do not have an aligned exact checkpoint. P6 finds their last
fully written snapshot and continues it with a fresh recorded RNG, labels the continuation
`statistical`, and preserves source-file provenance. Once that continuation completes, its own endpoint
is an exact checkpoint for subsequent segments. The source HDF5 is never appended to or overwritten.

## Initial concentrations and special-area accounting

`concentration_a` and `concentration_b` set the initial hydrogen percentage on the ordinary available
sites in the left and right halves. The spot and trap layer have independent site density and initial
occupancy settings:

```python
concentration_a = 50
concentration_b = 50
concentration_spot = 50  # Only used when USE_SPOT = True
concentration_trap_layer = 0  # Only used when USE_TRAP_LAYER = True

max_sol_spot = Fraction(100, 100)
TRAP_LAYER_CENTER_X = x // 2
TRAP_LAYER_WIDTH = 20
max_sol_trap_layer = Fraction(100, 100)
```

For a simulation whose approximate steady-state shape is already known, the two ordinary areas can
instead start with independent left-to-right linear concentration profiles:

```python
USE_INITIAL_CONCENTRATION_PROFILE = True
concentration_profile_a = (100, 50)  # Area A: 100% at x=0 to 50% at x=L/2
concentration_profile_b = (50, 0)    # Area B: 50% at x=L/2 to 0% at x=L
```

Each percentage describes the fraction of that column's available sites that initially contain H. The
assignment is rounded to a whole number of H atoms per column, so a sparse lattice follows the requested
line approximately rather than exactly. A flat profile such as `(40, 40)` is also valid. When
`USE_INITIAL_CONCENTRATION_PROFILE = False`, the original uniform `concentration_a` and
`concentration_b` settings are used.

`max_sol_spot` determines what fraction of pixels inside the circular spot become possible H sites;
`concentration_spot` determines what fraction of those sites initially contain H. For example,
`max_sol_spot = Fraction(20, 100)` and `concentration_spot = 50` creates possible sites on approximately
20% of the spot geometry and initially fills half of those sites.

`TRAP_LAYER_CENTER_X` moves the full-height vertical layer horizontally. The configured width is centred
on that column and is clipped if it reaches a matrix edge. `max_sol_trap_layer` likewise determines what
fraction of layer pixels become possible H sites; `concentration_trap_layer` determines what fraction of
those sites initially contain H.

Initialization first applies either the uniform bulk concentrations or these optional profiles. The
source/sink state is applied next. The trap layer then replaces its exact rectangle with its own
solubility and concentration. The spot is created last with `max_sol_spot` and
`concentration_spot`, so the spot owns every pixel where the two special areas overlap. If a movable
trap or spot overlaps a source/sink band, that later special area likewise owns the overlap.

The same exact masks are used by initialization, movement characteristics, movement-statistics regions,
and still-diagram calculations. The spot and trap labels use only their owned red and blue sites.
Left/right, source/sink, and custom rectangular annotations subtract whichever parts of the spot and trap
overlap their masks. The still-diagram concentration profile excludes the spot but deliberately retains
the full-height trap layer so its concentration remains visible in the profile. Purple pixels never enter
an occupancy denominator.

Consequently:

```text
spot concentration = red inside spot / (red + blue inside spot)
trap concentration = red inside trap but outside spot / (red + blue inside trap but outside spot)
bulk region concentration = red outside spot and trap / (red + blue outside spot and trap)
```

No assumption is made about which half contains the spot or trap layer; ownership and exclusion are
performed by exact mask intersection.

## Still-diagram visualization modes

`c3_Brown_Make_Diagram.py` can show an exact snapshot or derive a more legible abstraction from that
same snapshot. Choose a preset near the top of that file:

```python
DIAGRAM_PRESET = "default"
DIAGRAM_PRESET = "depletion_heatmap"
DIAGRAM_PRESET = "printer_friendly"
DIAGRAM_PRESET = "area_summary"
DIAGRAM_PRESET = "chapter_2_3_brown_overview"
```

Changing this setting does not alter or rerun the simulation. It only changes how one saved HDF5 frame
is drawn.

Still diagrams can be saved as PNG, PDF, or editable SVG near the top of
`c3_Brown_Make_Diagram.py`. SVG is the preferred format for further editing in Inkscape:

```python
SAVE_PNG = False
SAVE_PDF = False
SAVE_SVG = True
SVG_TEXT_AS_PATHS = False  # Keep titles and axis labels editable in Inkscape
```

`area_summary` preserves the general stylized-dot configuration, including its A/B average labels.
`chapter_2_3_brown_overview` follows the saved concentration profile across x, omits those
regional-average labels, and draws the source/sink side bands three times wider for emphasis.

The ordinary `pixels` and `dots` render modes retain individual matrix locations. `pixels` gives every
matrix cell one image pixel. `dots` draws every available site and H atom at its exact coordinate. A
large dot size can help individual points survive downscaling, but neighbouring markers then overlap.
Because H markers are drawn over available-site markers, a dense patch can look almost solid red even
when its occupancy is far below 100%. The `dots` mode is therefore useful for exact-coordinate views,
but increasing its marker size is not a reliable print abstraction.

### `depletion_heatmap`

The `depletion_heatmap` preset is intended to make the spot's depletion or accumulation halo visible.
It separately smooths the local H count and the local available-site count and then divides them:

```text
smoothed local occupancy = smoothed H count / smoothed available-site count
displayed deviation = smoothed local occupancy - bulk reference occupancy
```

This order matters when the spot and bulk have different solubilities. Smoothing the red/blue matrix
values directly would mix site density with H occupancy and create a misleading edge.

By default, blue means fewer occupied available sites than the corresponding bulk reference, white
means approximately the bulk value, and red means more. The A and B bulk references are calculated
after excluding the exact spot and trap masks, so a movable or partly overlapping spot cannot
contaminate either reference. Smoothing is also performed independently across special-area boundaries;
an empty high-solubility spot therefore does not manufacture a blue halo merely by being averaged into
the surrounding bulk. A halo outside the dashed spot outline comes from the saved simulation result.

The main controls in `depletion_heatmap.py` are:

```python
HEATMAP_SIGMA = 28.0                 # Spatial smoothing radius in matrix cells
HEATMAP_DEVIATION_LIMIT = 20.0       # Symmetric colour range in percentage points
HEATMAP_MODE = "deviation"           # Or "occupancy" for absolute local occupancy
HEATMAP_REFERENCE_MODE = "regional_bulk"  # Or "global_bulk"
```

The heatmap is a derived view, not an additional physical field and not a change to the movement
model. Fine spatial detail below the smoothing scale is deliberately suppressed.

### `printer_friendly`

The `printer_friendly` preset replaces overlapping particle markers with non-overlapping spatial-bin
glyphs. Each circle summarizes one local block of matrix cells:

```text
red sector of circle = H / (H + unoccupied available sites)
circle size          = density of available sites in that block
blue remainder       = unoccupied fraction of the available sites
```

Thus a half-red circle means 50% occupancy regardless of how densely that area provides sites. A larger
circle means the area has more possible H sites. This keeps concentration and solubility visually
separate, and a low-occupancy high-solubility spot can no longer look solid red because thousands of
large particle markers were painted on top of each other.

The glyphs keep approximate location at the bin scale, while intentionally discarding exact within-bin
coordinates. This makes them suitable for print and for qualitative communication, not for inspecting
individual jumps. Dashed outlines retain the spot and trap-layer geometry, and the legend reports their
exact red-over-red-plus-blue concentration from the underlying snapshot.

The main controls in `printer_friendly.py` are:

```python
GLYPH_BIN_SIZE = 32                  # Larger value = fewer, larger summary circles
GLYPH_MIN_RADIUS_FRACTION = 0.38
GLYPH_MAX_RADIUS_FRACTION = 0.90
GLYPH_CAPACITY_GAMMA = 0.30          # Compresses very large solubility differences
```

### `area_summary`

The `area_summary` preset keeps the familiar matrix geometry but reconstructs it as a clean illustrative
dot field. It does not plot the millions of literal matrix pixels or preserve individual H positions.
Instead, it measures each area's average occupancy from the selected snapshot, places a manageable
number of large dots on a synthetic staggered lattice inside that area's real shape, and colours the
same fraction red.

In the default `area_average` concentration mode, if Area A is 40% occupied, 40% of its displayed dots
are red and 60% are blue. Their colours are randomly mixed across the even positions rather than copied
from the simulation. The percentage still comes from the real result:

```text
area occupancy = H in owned area / all available sites in owned area
```

The same robust A/B, spot, and trap ownership masks are used as elsewhere: the spot owns its overlap,
then the trap layer, then A/B. With the default `available_sites` density mode, the number of illustrative
dots assigned to each area is proportional to its measured number of available sites. A 10%-solubility
area therefore appears about twice as densely dotted as an equally large 5%-solubility area, while the
red/blue ratio independently shows occupancy.

`even_hex` position mode fits a staggered lattice to every area. Areas A and B share the same lattice
phase and outer inset, so their synthetic dots continue naturally across x=L/2 without an artificial
empty separator. They still retain separate ownership and concentrations. A configurable minimum
centre-to-centre spacing prevents marker overlap, while the position inset keeps circles away from real
material outlines and the outside frame. Very dense or small areas, such as a 100%-solubility spot, may
not physically fit their proportional target number of print-sized dots. In that case the renderer caps
the area at its non-overlapping visual capacity but keeps the measured red/blue ratio exact. Dot density
therefore remains qualitative once this capacity is reached; it never uses overlap to imply extra
capacity.

An optional constrained shake is applied after the even lattice is fitted. Every proposed movement is
accepted only if the centre remains inside its owned area and at least
`AREA_SUMMARY_MIN_DOT_SPACING` from every other displayed centre. This introduces irregular gaps without
bringing back marker overlap. Four profiles are available:

```text
none       untouched staggered lattice
gentle     very small one-pass disturbance
organic    moderate irregularity without deliberate attraction
clustered  stronger disturbance plus attraction toward random local centres
```

`clustered` is allowed to create visible bunches and empty patches, but these are illustrative. They do
not claim that the simulation measured physical clustering. Use `none` when the most neutral encoding is
required and `organic` when a less mechanical presentation is preferred.

By default, one invocation renders both `organic` and `clustered`. When saving is enabled, the filenames
receive `_organic` and `_clustered` suffixes so neither output overwrites the other. The output list can
be reduced or replaced in the preset:

```python
AREA_SUMMARY_SHAKE_OUTPUT_MODES = ("organic", "clustered")  # Default: render both
AREA_SUMMARY_SHAKE_OUTPUT_MODES = ("organic",)              # Render only organic
AREA_SUMMARY_SHAKE_OUTPUT_MODES = ("none", "gentle")        # Any desired combination

# Single-output form: no style suffix is added to the filename.
AREA_SUMMARY_SHAKE_OUTPUT_MODES = None
AREA_SUMMARY_SHAKE_MODE = "organic"
```

The clustered layout also has two geometry scopes. `combined_a_b` creates one set of attraction centres,
combines all A+B dots, and shakes them in one pass inside the union of the two base geometries. The
placement algorithm therefore has no boundary at x=L/2, and a dot may cross that bookkeeping boundary
during the decorative shake. Its final side and x-slice determine how it is subsequently coloured.
The total number of base dots is preserved, although clustering intentionally allows their local density
to shift. The spot and trap layer remain separate because they have independent concentrations and site
densities. Use `per_area` when preserving strict A/B positional ownership is more important than making
one continuous clustered field; it gives A and B unrelated movement passes and cluster centres:

```python
AREA_SUMMARY_CLUSTER_SCOPE = "combined_a_b"  # Default
AREA_SUMMARY_CLUSTER_SCOPE = "per_area"
```

For transient states, the red/blue assignment can vary from left to right while keeping the same clean
dot layout. Every displayed position remains a discrete state: it is either red (occupied) or blue
(empty). The profile determines the exact rounded number of red dots in each narrow vertical slice, and
those red dots are randomly mixed within the slice so the result does not become a rigid striped chart.

`saved_x_profile` measures the selected HDF5 snapshot column-wise, excludes the independently owned spot
and trap from the A/B profile, smooths H counts and available-site counts, and uses that result for the
slice totals:

```python
AREA_SUMMARY_CONCENTRATION_MODE = "saved_x_profile"
AREA_SUMMARY_X_PROFILE_SIGMA = 20.0
AREA_SUMMARY_CONCENTRATION_BIN_WIDTH = 40
```

`linear_x` supplies an illustrative profile directly. For example, this produces 100% red at the left
edge, 0% red at the right edge, and the appropriate discrete mixture between them:

```python
AREA_SUMMARY_CONCENTRATION_MODE = "linear_x"
AREA_SUMMARY_LINEAR_CONCENTRATION = (100, 0)
AREA_SUMMARY_CONCENTRATION_BIN_WIDTH = 40
```

The spot and trap always keep their own measured area-average concentrations in these transient modes.
Use a smaller bin width for closer local agreement and more visible statistical texture, or a larger
width for a smoother-looking illustration. Set `AREA_SUMMARY_CONCENTRATION_MODE = "area_average"` to
restore the steady-state A/B summary.

Source and sink strips are drawn as solid red and blue boundary bands. `area_average` retains the full
measured regional averages used by the ordinary diagram, including those boundary conditions.
`saved_x_profile` likewise measures them in their real x positions; `linear_x` instead follows the
explicitly configured illustrative endpoints.

The reconstruction is deterministic: the configured seed produces the same lattice phase and random
colour arrangement each time, making a result reproducible. Changing the seed changes only the
illustrative arrangement, not any measured percentage. The older uniform-random position style remains
available as `AREA_SUMMARY_POSITION_MODE = "random"`, but `even_hex` is the print-oriented default.

This is a presentation abstraction. It is well suited to steady-state or final comparison figures, but
it deliberately hides within-area gradients, diffusion fronts, and depletion halos. Use
`depletion_heatmap`, `printer_friendly`, or an exact snapshot when those spatial details matter.

The main controls in `area_summary.py` are:

```python
AREA_SUMMARY_TOTAL_DOTS = 5000
AREA_SUMMARY_DENSITY_MODE = "available_sites"  # Or "uniform_area"
AREA_SUMMARY_RANDOM_SEED = 104729
AREA_SUMMARY_CONCENTRATION_MODE = "area_average"  # area_average/saved_x_profile/linear_x
AREA_SUMMARY_LINEAR_CONCENTRATION = (100, 0)
AREA_SUMMARY_X_PROFILE_SIGMA = 20.0
AREA_SUMMARY_CONCENTRATION_BIN_WIDTH = 40
AREA_SUMMARY_POSITION_MODE = "even_hex"         # Or "random"
AREA_SUMMARY_MIN_DOT_SPACING = 8.0
AREA_SUMMARY_SHAKE_OUTPUT_MODES = ("organic", "clustered")
AREA_SUMMARY_SHAKE_MODE = "organic"             # Fallback when output modes is None
AREA_SUMMARY_SHAKE_STRENGTH = None              # None uses the selected profile
AREA_SUMMARY_SHAKE_PASSES = None
AREA_SUMMARY_CLUSTER_SCOPE = "combined_a_b"      # Or "per_area"
AREA_SUMMARY_POSITION_INSET = 5
AREA_SUMMARY_DOT_SIZE = 18
AREA_SUMMARY_SHOW_SOURCE_SINK_BANDS = True
AREA_SUMMARY_SOURCE_SINK_BAND_WIDTH_SCALE = 1.0
```

## Area characteristics, affinity, and mobility

The user-facing physical model currently has four area characteristics:

```text
a           left base material
b           right base material
    trap_layer  optional movable full-height vertical layer
spot        optional circular area
```

These are configured area-wise, not pixel by pixel. Internally P6 compiles the geometry into a small
characteristic-ID map so the movement kernels can perform fast lookups. A/B are assigned first, the trap
layer overrides A/B when enabled, and the spot overrides every area it overlaps.

```python
AREA_CHARACTERISTICS = {
    "a": {"affinity": 1.0, "mobility": 1.0},
    "b": {"affinity": 1.0, "mobility": 1.0},
    "spot": {"affinity": 1.0, "mobility": 1.0},
    "trap_layer": {"affinity": 10.0, "mobility": 1.0},
}
```

`affinity` controls equilibrium preference; only ratios matter. P6 uses a Metropolis-style directed
acceptance rule. Moving toward an equal- or higher-affinity area keeps the ordinary distance-dependent
rate. Moving toward a lower-affinity area reduces that rate by:

```text
target affinity / source affinity
```

With the default trap-layer affinity of 10:

```text
A -> trap       ordinary rate
trap -> A       ordinary rate / 10
A -> A          ordinary rate
trap -> trap    ordinary rate
```

The trap can therefore retain H without slowing diffusion between points inside the trap.

With single-occupancy exclusion, affinity controls the equilibrium occupancy odds:

```text
[trap occupancy / (1 - trap occupancy)]
------------------------------------------------ = trap affinity / A affinity
[A occupancy / (1 - A occupancy)]
```

At low occupancy this is approximately the ordinary occupancy ratio. At high occupancy, saturation
matters. Physical H per matrix area additionally includes each area's available-site density.

`mobility` is a separate symmetric kinetic scale from 0 to 1. An edge uses the geometric mean of its two
areas' mobilities. Lower mobility slows both directions equally and therefore changes kinetics without
changing the affinity ratio. `base_movement_probability` remains a global kinetic scale; changing it
uniformly does not create an equilibrium preference.

The same area transition table is used by `molecular_wiggle`, `random_sequential_wiggle`, and
`event_driven_wiggle`. The event-driven scheduler uses uniformized steps so that its saved configurations
retain the residence preference encoded by those directed rates while null waiting is skipped
computationally.

## `molecular_wiggle`

This is the direct, synchronous reference model and the closest implementation of explicit molecular
wiggles. During one step, every hydrogen atom proposes one random
X/Y displacement. The displacement is accepted according to the Gaussian distance probability. All
proposals inspect the state at the beginning of the step; when several atoms request one empty site, a
random priority selects the winner. Accepted winners are applied together.

One step means one proposal opportunity for every hydrogen atom present at the beginning of the step.
Its synchronous update rule is still a modelling choice, so "reference model" is more accurate than
claiming that it is uniquely the most physically realistic mode.

## `random_sequential_wiggle`

Aside from some memory optimization the idea is to remove the 99.4% rejected moves (wiggling but staying
put). this is done by removing the lowest present propability of not moving by dividing by it, this
keeps the relative likelihoods of movement but removes most of the non-movement overhead. For example
Particle A moves 1% of the time, B 2% of the time. 2/2 = 1, B moves everytime it gets selected, 1/2= 0.5,
A moves half the time it gets selected. 
This mode uses the same single-hydrogen wiggle probabilities as `molecular_wiggle`, with an asynchronous
sequential update rule. During startup, P6 precomputes the marginal probability of every reachable
source-site to destination-site wiggle. All random coordinates that would miss a site, leave the matrix, 
or fail the Gaussian test are represented by the remaining probability of staying in place.

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

## `event_driven_wiggle`

This is the uniformized, rate-weighted version of `random_sequential_wiggle`. It uses the same
precomputed source-to-destination probabilities, compact site states, dynamic hydrogen list, exclusion
check, and source/sink behavior.

Each hydrogen carries its total destination-proposal probability. A dynamic partial-sum tree stores
those weights. P6 mathematically embeds them in a constant-rate sequence of uniformized opportunities.
Most opportunities would change nothing; instead of executing them one by one, P6 samples their
geometric waiting length and jumps directly to the next actual proposal. It then selects a hydrogen in
proportion to its total directed rate and a destination in proportion to that transition's rate.

One step means one uniformized wiggle opportunity. Null opportunities are counted but skipped in one
operation, so `steps` and `save_every_steps` describe the common uniformized clock rather than the number
of actual proposals. HDF5 stores `proposal_event_count` separately. A proposed occupied destination
still changes no state.

Uniformization preserves the affinity-dependent residence distribution that would be lost if only
actual proposals were counted. The clock remains dimensionless until an attempt frequency or physical
rate scale is supplied, but its relative waiting and equilibrium behavior are part of the model.

## Deprecated: `forced_jump`

This deprecated heuristic is retained only so old comparisons remain runnable. It does not use
`AREA_CHARACTERISTICS`. It selects from currently valid available destination sites
within the configured radius. If a valid destination exists, the hydrogen jumps without the Gaussian
proposal/rejection behavior used by the wiggle-derived modes. Its compact precomputed lane is used when
sink/source boundaries are disabled; a matrix-scanning safe lane is used when those boundaries can
change the hydrogen population.

One step is one forced-jump update sweep. Because Gaussian distance weights and no-move probabilities
are omitted, this mode is not physically equivalent to the wiggle-derived modes.

## How the wiggle modes relate physically

The three wiggle-derived modes share the same basic microscopic picture: hydrogen occupies discrete
sites, a source site has distance-weighted possible destinations, and a destination must be empty before
the hydrogen can occupy it. The differences are mainly about update order and how the waiting between
movement proposals is represented.

That makes the modes closely related, but not completely interchangeable. They use the same spatial
movement law while making different choices about kinetics.

### A proposal is not always a successful move

Suppose a hydrogen has five structurally possible destinations and each occupies a probability interval
of 0.01. A random value below 0.05 selects one of those destinations; a value above 0.05 selects no
destination. After a destination is selected, the move can still fail if another hydrogen currently
occupies that site.

The precomputed transition lookup contains structurally possible sites, including sites that may
currently contain hydrogen. Occupancy changes after every successful sequential move, so it is checked
dynamically rather than built into the static transition table.

This gives two distinct kinds of unchanged state:

- No destination was proposed because the random value fell outside the transition CDF.
- A destination was proposed, but it was occupied.

`event_driven_wiggle` skips runs of the first kind computationally but retains their count through
uniformization. It deliberately retains the second kind directly and therefore preserves
hydrogen-exclusion behavior.

### Synchronous and sequential updates

`molecular_wiggle` and `random_sequential_wiggle` use the same single-hydrogen displacement
probabilities, but they do not use the same update schedule.

In `molecular_wiggle`, every hydrogen present at the beginning of a sweep receives exactly one
opportunity. Every proposal sees the same beginning-of-sweep state, and conflicting claims are resolved
before the winners are applied together.

In `random_sequential_wiggle`, a sweep consists of as many random selections as there were hydrogen
atoms at its start. Selection is with replacement: some atoms may be selected zero times, some once, and
some more than once. Successful moves are applied immediately, so later selections see the updated
state.

This is a genuine modelling distinction rather than only a programming optimization. Molecules do not
physically move in synchronized sweeps, so asynchronous updates can be a natural model as well. On the
other hand, the synchronous sweep is a direct and easily inspected implementation of the original
"every atom wiggles once" idea.

When proposal probabilities are small and the system is not so crowded that simultaneous interactions
are common, the difference becomes small. Conflicts and multiple consequential selections are then
higher-order events. In that low-probability or dilute limit, the synchronous and sequential models
approach the same continuous-time behavior.

### Why weighted event selection plus uniformization preserves the process

Let hydrogen number `i` have total destination-proposal probability `r_i`. With `N` hydrogen atoms,
`random_sequential_wiggle` first selects an atom uniformly, so the probability that one ordinary attempt
produces a proposal from atom `i` is:

```text
r_i / N
```

If `R` is the sum of all hydrogen proposal probabilities, the probability that an ordinary attempt
produces any proposal is:

```text
R / N
```

Conditioned on the fact that a proposal occurred, the probability that atom `i` produced it is therefore:

```text
(r_i / N) / (R / N) = r_i / R
```

That is exactly the weighted selection used by `event_driven_wiggle`. After selecting the hydrogen, the
mode selects its destination in proportion to the individual transition probabilities. It is not
inventing a different movement preference; it samples directly from the proposals that the sequential
mode would eventually produce.

For example, consider two hydrogen atoms:

- Atom A has a proposal probability of 1%.
- Atom B has a proposal probability of 2%.

An ordinary uniform attempt produces an A proposal with probability 0.5%, a B proposal with probability
1%, and no proposal with probability 98.5%. Among actual proposals, A is responsible for one third and B
for two thirds. The event-driven mode selects them with exactly those weights, while its sampled
geometric waiting length accounts for the 98.5% null opportunities.

### Why the skipped waiting still matters

Consider one hydrogen moving between two positions:

- A slow or trap position has a 1% proposal probability.
- A fast position has a 10% proposal probability.

With attempt-based time, the hydrogen waits about 100 opportunities in the trap and about 10 in the fast
position. An observation made at a random opportunity therefore finds it in the trap roughly 91% of the
time. If all waiting were removed, the event sequence could simply alternate and show each position
about half the time.

The current implementation does not remove that waiting from the stochastic model. It draws the number
of uniformized null opportunities in one operation, advances the step counter by that number, and then
performs the weighted proposal. This keeps the 91/9 residence behavior without paying for each null
opportunity individually.

### How modes 3 and 4 relate in time

The geometric waiting used here is the discrete uniformized counterpart of kinetic Monte Carlo waiting.
It gives `random_sequential_wiggle` and `event_driven_wiggle` compatible directed-rate equilibrium and
relative residence behavior, while their step units remain different. An exponential waiting time would
be the usual continuous-time kinetic Monte Carlo formulation.

Neither a sweep nor an attempt has physical units until a material- and temperature-dependent attempt
frequency or comparable rate information is supplied. `random_sequential_wiggle` retains the explicit
per-H attempt and population accounting needed for such a later calibration. The event-driven mode
retains a dimensionless uniformized clock and separately records how many actual proposals occurred.

### Practical interpretation

| Mode | What is preserved | What its step means | Useful interpretation |
|---|---|---|---|
| `molecular_wiggle` | Explicit displacement proposals and synchronous conflicts | One global sweep | Direct synchronous reference model |
| `random_sequential_wiggle` | Single-H probabilities, no-move attempts, and asynchronous state changes | One average opportunity per H | Attempt-based kinetics and possible later time calibration |
| `event_driven_wiggle` | Directed rates, null residence, and weighted proposals | One uniformized opportunity | Fast area-aware kinetics with a dimensionless clock |
| `forced_jump` (deprecated) | Only currently valid destination geometry | One forced-jump sweep | Legacy heuristic comparison only |

Equal step counts should not be compared across these modes. Even equal numbers of successful moves are
not always equal diffusion, because jump-distance distributions, update order, and waiting behavior can
differ. Concentration profiles and qualitative spatial evolution can still be compared, provided the
chosen step convention is stated clearly.

### Representative performance and movement comparison

The following benchmark used the same representative matrix for every mode: 47,008 possible sites,
20,995 hydrogen atoms, and a jump radius of 10. Compilation, lookup construction, snapshot reconstruction,
and HDF5 writing were excluded. Runtime depends on the computer and current system load, so these values
are an example rather than a guaranteed speed.

| Mode | Time per step or event | Successful H moves per step or event | Successful H moves per second |
|---|---:|---:|---:|
| `molecular_wiggle` | 0.4646 ms per sweep | 93.27 | 0.201 million |
| `random_sequential_wiggle` | 0.0749 ms per sweep | 93.13 | 1.24 million |
| `event_driven_wiggle` | Rebenchmark required after uniformization | Depends on current total rate | Rebenchmark required |
| `forced_jump` (deprecated) | 6.8654 ms per sweep | 19,848 | 2.89 million |

`molecular_wiggle` and `random_sequential_wiggle` produced almost exactly the same number of successful
moves per sweep in this test. The sequential implementation calculated them about 6.2 times faster;
repeated benchmarks have placed this gain in the approximate range of 6 to 7 times.

The 6.2-times molecular-versus-random-sequential comparison remains representative because those two
step definitions did not change.

An earlier raw-proposal version of `event_driven_wiggle` measured 246 ns per actual proposal, with 56.3%
of proposals becoming successful moves. Those figures are useful as historical kernel measurements, but
they are not a benchmark for the current uniformized step definition. The current cost and movement per
step depend on the ratio between the population's total directed rate and the uniformization bound, so
the area-aware implementation must be benchmarked separately.

The deprecated forced-jump numbers need a different interpretation. One forced sweep moved almost every hydrogen,
which is why it produced nearly 20,000 relocations at once. Those relocations ignore Gaussian/no-move
probabilities and have a different jump-distance distribution. Its high movement throughput is useful
for a heuristic comparison, but matching only the number of moves does not make it physically equivalent
to the wiggle-derived modes.

## Reproducibility

All modes record the actual random seed and algorithm in HDF5 metadata. Set `random_seed` to an integer
to reproduce initialization and movement. Leave it as `None` to create and record a fresh seed per run.
