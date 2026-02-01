"""
This is just to comfortably start the different simulations
"""

import sys
import subprocess
import io
import contextlib

# Suppress config prints during import, so no double prints. They’ll still appear when scripts run directly.
with contextlib.redirect_stdout(io.StringIO()):
    from b4_functions import get_value


def run_script(script_name):
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {script_name}: {e}")


if __name__ == "__main__":

    simulation = "weld"  # weld | microstructure | simple_heat_diffusion
    simulation_type = get_value("simulation_type")  # lap joint | butt joint | iso3690 - in b2_param_config.py

    if simulation == "weld":
        # Run the main simulation script
        run_script('b1_Main_WeldCraft.py')

        # Run the script that makes you a nice animation of it
        if simulation_type == "iso3690":
            run_script('c2_Make_Animation_ISO3690.py')
        else:
            run_script('c1_Make_Animation.py')

    if simulation == "microstructure":  # Needs work, not sure it works after all the logic reworks
        # Run the main simulation script
        run_script('micro_structure_diffusion.py')

        # Run the script that makes you a nice animation of it
        run_script('c10_Micro_Make_Animation.py')
