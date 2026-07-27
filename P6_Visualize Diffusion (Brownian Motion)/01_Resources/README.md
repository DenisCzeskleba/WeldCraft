# P6 Brownian Motion User Guide

This resource README describes the simulation choices available to P6 users. It is separate from the
WeldCraft repository README.

P6 provides four movement modes. They share the same matrix initialization, concentration settings,
spot/layer topology, HDF5 snapshot format, and plotting tools. They differ in how molecular movement is
executed.

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

This is the rate-weighted proposal-event version of `random_sequential_wiggle`. It uses the same
precomputed source-to-destination probabilities, compact site states, dynamic hydrogen list, exclusion
check, and source/sink behavior. The difference is that it does not execute selections whose random
wiggle would select no destination.

Each hydrogen carries its total destination-proposal probability. A dynamic partial-sum tree stores
those weights. For every event, P6 selects a hydrogen in proportion to its total probability and then
selects one of that hydrogen's destinations in proportion to the individual transition probabilities.
Consequently, a hydrogen in a low-mobility or trap site appears less often than one in a high-mobility
site. A selected destination can still be occupied, in which case the proposal event changes no state.

One step in this mode means one weighted destination-proposal event. `steps = 20_000_000` therefore
means 20 million proposal events, and `save_every_steps = 25000` saves every 25,000 events. These values
are deliberately not comparable with the attempt-based steps of `random_sequential_wiggle` or the
synchronous sweeps of `molecular_wiggle`.

This mode preserves the conditional proposal-event law but removes state-dependent waiting between
events. It is intended for fast qualitative evolution. Event number is not a physical time axis and does
not preserve quantitative residence times or diffusion rates. If physical rate data becomes available,
use `random_sequential_wiggle` for its explicit attempt accounting, or extend the event-driven mode with
a kinetic-time calculation.

## `forced_jump`

This is a heuristic high-mobility mode. It selects from currently valid available destination sites
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

`event_driven_wiggle` removes the first kind. It deliberately retains the second kind and therefore
preserves hydrogen-exclusion behavior.

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

### Why weighted event selection reproduces the proposal chain

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
mode would eventually produce after its no-destination attempts are removed.

For example, consider two hydrogen atoms:

- Atom A has a proposal probability of 1%.
- Atom B has a proposal probability of 2%.

An ordinary uniform attempt produces an A proposal with probability 0.5%, a B proposal with probability
1%, and no proposal with probability 98.5%. Among the proposals that remain, A is responsible for one
third and B for two thirds. The event-driven mode selects them with exactly those weights.

### The missing piece is residence time

Removing no-proposal attempts preserves the conditional proposal sequence, but it does not preserve how
long the system waits in each configuration when event number is used as the horizontal axis.

Consider one hydrogen moving between two positions:

- A slow or trap position has a 1% proposal probability.
- A fast position has a 10% proposal probability.

With attempt-based time, the hydrogen waits about 100 attempts in the trap and about 10 attempts in the
fast position. An observation made at a random attempt therefore finds it in the trap roughly 91% of the
time. If all waiting is removed, the event sequence can simply alternate between trap and fast position;
frames saved after every event would show each position about half the time.

With many hydrogen atoms the effect is less extreme because other atoms continue to produce events while
one atom is trapped. The trapped atom is still selected less frequently relative to the others. Even so,
the total event rate can change as the population moves between microstructures, traps, sources, and
sinks. Event count is therefore not generally proportional to physical time.

If every configuration had the same total proposal rate, removing the waiting would amount to a constant
rescaling of time. When the total rate changes with the configuration, it is not a single constant
rescaling.

### When modes 3 and 4 can describe the same time-dependent physics

An event-driven method can retain physical kinetics without executing every rejected attempt. After each
weighted proposal it can generate a waiting time from the current total rate. A geometric waiting time
would reproduce the present discrete attempt model; an exponential waiting time would give the usual
continuous-time kinetic Monte Carlo formulation.

With that waiting-time calculation included, `random_sequential_wiggle` and the event-driven algorithm
would be two mathematical implementations of essentially the same time-dependent stochastic process.
The current `event_driven_wiggle` intentionally omits that clock because it is intended for rapid
qualitative evolution.

Neither a sweep nor an attempt has physical units until a material- and temperature-dependent attempt
frequency or comparable rate information is supplied. `random_sequential_wiggle` retains the explicit
attempt and population accounting needed for such a later calibration. The current event-driven mode
retains the correct proposal-event ordering but would need kinetic waiting times before event number
could be interpreted quantitatively.

### Practical interpretation

| Mode | What is preserved | What its step means | Useful interpretation |
|---|---|---|---|
| `molecular_wiggle` | Explicit displacement proposals and synchronous conflicts | One global sweep | Direct synchronous reference model |
| `random_sequential_wiggle` | Single-H probabilities, no-move attempts, and asynchronous state changes | One average opportunity per H | Attempt-based kinetics and possible later time calibration |
| `event_driven_wiggle` | The weighted sequence of destination proposals | One proposal event | Fast qualitative evolution without a physical event clock |
| `forced_jump` | Only currently valid destination geometry | One forced-jump sweep | Heuristic high-mobility comparison |

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
| `event_driven_wiggle` | 246 ns per event | 0.563 | 2.29 million |
| `forced_jump` | 6.8654 ms per sweep | 19,848 | 2.89 million |

`molecular_wiggle` and `random_sequential_wiggle` produced almost exactly the same number of successful
moves per sweep in this test. The sequential implementation calculated them about 6.2 times faster;
repeated benchmarks have placed this gain in the approximate range of 6 to 7 times.

One event-driven proposal became a successful move about 56.3% of the time. Matching the approximately
93 successful moves in one wiggle sweep therefore requires roughly:

```text
93 / 0.563 = 165 event-driven events
```

Those 165 events took about 0.0406 ms. At equal successful-movement count, the representative comparison
was therefore:

- `random_sequential_wiggle` was about 6.2 times faster than `molecular_wiggle`.
- `event_driven_wiggle` was about 11.4 times faster than `molecular_wiggle`.
- `event_driven_wiggle` was about 1.8 times faster than `random_sequential_wiggle`.

This also explains why 20 million event-driven steps are not comparable with 20 million wiggle sweeps.
At the measured rates, 20 million event-driven events produce about 11.3 million successful moves,
equivalent to only about 121,000 wiggle sweeps. Matching the successful-movement count of 20 million
wiggle sweeps would require approximately 3.3 billion event-driven events. The corresponding pure-kernel
estimates on the benchmark computer were roughly 155 minutes for `molecular_wiggle`, 25 minutes for
`random_sequential_wiggle`, and 13 to 14 minutes for an equal-movement event-driven run.

The forced-jump numbers need a different interpretation. One forced sweep moved almost every hydrogen,
which is why it produced nearly 20,000 relocations at once. Those relocations ignore Gaussian/no-move
probabilities and have a different jump-distance distribution. Its high movement throughput is useful
for a heuristic comparison, but matching only the number of moves does not make it physically equivalent
to the wiggle-derived modes.

## Reproducibility

All modes record the actual random seed and algorithm in HDF5 metadata. Set `random_seed` to an integer
to reproduce initialization and movement. Leave it as `None` to create and record a fresh seed per run.
