# WeldCraft ![WIP Badge](https://img.shields.io/badge/status-WIP-yellow.svg) <a href="https://doi.org/10.5281/zenodo.18451838"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.18451838.svg" alt="DOI"></a> <a href="https://www.buymeacoffee.com/DenisCzeskleba"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" height="20px"></a> 

Open-Source repository for all (currently work in progress) hydrogen/heat weld simulation scripts and programs relating to the research project "Prevention of Cold Cracking in Thick Plated High-Strength Steel SAW-Welds"

## Status - v0.4.1

CURRENT STATE (July 2026): Another total script refractor.

This is a, work-in-progress release. It's meant to get started and provide a basic framework, but it's not feature-complete, and the code is subject to change. The structure and comments are minimal, and functionality will be updated regularly. Use at your own risk. 

Until at least Q4 of 2026, this project will see constant updates, either here or on my local machine. Version 1.0 is expected by the end of Q4 2026.

## Modules

| Module | Name | Purpose |
| --- | --- | --- |
| P0 | WeldCraft Launcher | Starts launcher-enabled WeldCraft programs and provides module descriptions. |
| P1 | Simulate Hydrogen Diffusion | Interactive 1D/2D hydrogen-diffusion and heat-transport simulation with numerical and animation tools. |
| P2 | Hydrogen Diffusion During Welding | Full code-driven thermal and hydrogen-diffusion welding simulation, including thermal calibration workflows. |
| P3 | Heat Map | Focused thermal-welding simulation for manual heating calibration, four-point temperature traces, animations, and a practical smaller-scale example of P2 thermal concepts. |
| P4 | Hydrogen Permeation Atlas | Generates normalized 1D permeation-response diagrams for ideal diffusion, changing entry conditions, McNabb-Foster trapping, and aged prefilling. |
| P5 | Lattice Visualizer | Interactive SC, BCC, and FCC lattice visualization with configurable dopants and overlays. |
| P6 | Visualize Diffusion (Brownian Motion) | GUI and code-driven lattice-scale diffusion simulation, visualization, and analysis. |
| P7 | Analysis Tools | Reserved collection for additional WeldCraft analysis utilities. |

## License

This repository is licensed under the MIT License. 
You are free to use, modify, distribute, or even commercialize it under that license.

So feel free to use it which ever way you like, expand on it, change it, make it your own. 
If you find it helpful, please cite the project using the Zenodo DOI above or the metadata in `CITATION.cff`.

For more information, see the [LICENSE](LICENSE) file.

## How to cite

If you like and use this software, please cite the specific version you used.

**WeldCraft v0.4.1**

DOI: https://doi.org/10.5281/zenodo.18451838

The Zenodo record provides ready-to-import citation formats
(RIS, BibTeX, EndNote XML, etc.).
