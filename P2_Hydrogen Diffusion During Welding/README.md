# P2 Hydrogen Diffusion During Welding

Module-specific notes will be written here soon.

## Cooling curve exploration

`03_CodeBase/d3_fit_and_extrapolate_cooling_curve.py` is a standalone,
code-only analysis of measured cooling data. It reads the local measurement
files in `01_Resources/Cooling Curve Fit`, fits empirical cooling curves, and
compares an extrapolated high-temperature curve with a simple convection
estimate. It prints the fitted coefficients and writes its comparison figure
to `02_Results/04_Diagrams/Cooling Curve Fit`.

This script is exploratory; it does not automatically change P2 parameters or
calibrate the solver.
