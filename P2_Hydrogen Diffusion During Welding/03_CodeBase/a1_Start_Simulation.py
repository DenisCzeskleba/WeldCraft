"""
This is just to comfortably start the different simulations
"""

import sys
import subprocess
import io
import contextlib

# Suppress config prints during import, so no double prints. They'll still appear when scripts run directly.
with contextlib.redirect_stdout(io.StringIO()):
    from b3_Functions import get_value


def run_script(script_name):
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {script_name}: {e}")


if __name__ == "__main__":

    with contextlib.redirect_stdout(io.StringIO()):
        simulation_type = get_value("simulation_type")  # lap joint | butt joint | iso3690 - in b2_Simulation_Settings.py
        thermal_calibration = get_value("thermal_diffusion_calibration")  # set in b2_Simulation_Settings.py
        if thermal_calibration:  # Look up if you do normal simulation or a calibration run
            simulation = "thermal_diffusion_calibration"  # weld | "thermal_diffusion_calibration"
        else:
            simulation = "weld"  # weld | "thermal_diffusion_calibration"

    if simulation == "weld":
        # Run the main simulation script
        run_script('b1_Main_WeldCraft.py')

        # Run the script that makes you a nice animation of it
        if simulation_type == "iso3690":
            run_script('d2_Make_Animation_ISO3690.py')
        else:
            run_script('d1_Make_Animation.py')

    if simulation == "thermal_diffusion_calibration":  # Needs work!
        # Run the main simulation script
        run_script('b1_Main_WeldCraft.py')

        # Run the script that makes you a nice animation of it
        run_script('e2_Temperature_Calibration_4_Point_Output.py')
