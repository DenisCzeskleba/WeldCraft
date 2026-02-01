FILE NAMING AND FOLDER STRUCTURE
================================

This project follows a structured naming convention to keep all scripts organized
and logically ordered by their role in the workflow. Each script begins with a
letter prefix (a, b, c, d, e, i, ...) that reflects its place in the overall process.

Alphabetical sorting corresponds directly to execution order — from starting a
simulation to analyzing results and verifying solver consistency.

NAMING PATTERN
--------------

Files follow the pattern:

    <prefix><index>_<short_description>.py

Examples:

    a1_start_simulation.py
    b1_main_weldcraft.py
    b2_param_config.py
    c1_generate_animation.py
    d1_weld_diffusion_convergence_evaluation.py

Each script begins with a clear docstring describing its purpose, inputs/outputs,
and relationship to other parts of the workflow.


USAGE FLOW
-----------

Typical workflow order:

    A → B → C → D → E

Meaning:

    A : Launch the simulation (single or batch)
    B : Core computation (initialization, simulation, shared logic)
    C : Analyze results and generate plots or videos
    D : Evaluate numerical stability, accuracy, and convergence
    E : Provide information, examples, and documentation

This naming scheme ensures a clean, scalable structure and makes it clear how
each script contributes to the overall simulation and analysis pipeline.


OVERVIEW
--------

Prefix | Category                      | Description
-------+--------------------------------+----------------------------------------------------------
A      | Launchers / Starters           | Scripts that start the simulation process. These are
       |                                | lightweight entry points that run single simulations
       |                                | (a1) or batch runs (a2). No physical models or data
       |                                | handling occur here.
       |                                |
       |  a1_start_simulation.py        | Provides a simple, user-friendly way to start a single
       |                                | simulation. Depending on the selected mode (e.g. weld or
       |                                | microstructure), it launches the main solver and then
       |                                | the corresponding animation script automatically.
       |                                |
       |  a2_start_simulation_batch.py  | Automates a full parameter sweep by modifying
       |                                | b2_param_config.py for each set of parameters and
       |                                | launching multiple simulations in sequence. Each run
       |                                | saves its own output and animation in a dedicated batch
       |                                | folder. Includes automatic backup and restoration of
       |                                | the parameter file to prevent data loss.
B      | Core Simulation & Setup        | Main numerical solvers and configuration scripts:
       |                                |
       |  b1_main_weldcraft.py          | Central simulation driver for the thermal–hydrogen
       |                                | diffusion model. Handles the entire simulation workflow
       |                                | across all phases: welding, cooling, and post-weld
       |                                | diffusion. Manages geometry setup, boundary conditions,
       |                                | time stepping, data storage, and HDF5 snapshot saving.
       |                                | Integrates logic from b2_param_config.py,
       |                                | b3_initialization.py, and b4_functions.py.
       |                                |
       |  b2_param_config.py            | Primary user-facing configuration file. All adjustable
       |                                | simulation parameters, material constants, welding setup,
       |                                | diffusion coefficients, and output options are defined
       |                                | here. Users control nearly every aspect of the simulation
       |                                | from this script — it acts as the central tuning interface.
       |                                |
       |  b3_initialization.py          | Handles the spatial and physical setup of the simulation
       |                                | domain. Constructs the weld geometry, defines matrix
       |                                | dimensions (temperature, hydrogen, diffusion fields),
       |                                | and applies initial and boundary conditions for all
       |                                | supported configurations (butt joint, lap joint, ISO 3690).
       |                                | Also provides helper routines to generate weld timing steps
       |                                | and reset initial temperature distributions.
       |                                |
       |  b4_functions.py               | Central function library containing all reusable logic
       |                                | required by the simulation. Includes solvers, numerical
       |                                | update routines, interpolation helpers, boundary condition
       |                                | handling, hydrogen transport and diffusion utilities, as
       |                                | well as shared mathematical and I/O helpers. Designed to
       |                                | keep the main script readable while collecting all core
       |                                | functionality in one place.
C      | Postprocessing & Visualization | Scripts for processing and analyzing results, generating
       |                                | figures, videos, and data comparisons.
D      | Meta-Analysis / Verification   | Scripts for analyzing the simulation behavior itself,
       |                                | such as grid convergence, consistency checks, or
       |                                | verification of numerical accuracy.
       |                                |
       |  d1_weld_diffusion_convergence_evaluation.py
       |                                | Evaluates grid convergence and numerical consistency
       |                                | of weld diffusion simulations. Compares simulation runs
       |                                | with different spatial resolutions, computes L2/L∞ error
       |                                | norms, extracts probe data, estimates observed convergence
       |                                | order, and generates CSV summaries and diagnostic plots.
E      | Information & Examples         | Informational and documentation-related scripts and notes.
       |                                | These files do not perform calculations directly but serve
       |                                | to explain usage, workflows, and concepts behind the
       |                                | simulation suite. Eventually this category may also include
       |                                | the manual and example studies for demonstration or teaching.
       |                                |
       |  i99_example_workflow.py       | A descriptive example outlining a practical workflow for
       |                                | using the simulation suite to study hydrogen diffusion and
       |                                | welding parameters. Serves as an early draft of a user guide,
       |                                | explaining typical steps (geometry simplification, process
       |                                | calibration, parameter adjustment) and key physical
       |                                | considerations (t8/5, DHT, interpass temperature, convection,
       |                                | etc.). Text-based only — non-executable.
