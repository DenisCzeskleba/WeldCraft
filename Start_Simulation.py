"""
This is just to comfortably start the different simulations

"""

import sys
import subprocess


def run_script(script_name):
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {script_name}: {e}")


if __name__ == "__main__":

    # You can either do the "weld" simulation or "microstructure" from here
    simulation = "weld"  # weld | microstructure
    # You can also run simple_heat_diffusion extra!

    if simulation == "weld":
        # Run the main simulation script
        run_script('Heat_PDE_LDDE.py')

        # Run the script that makes you a nice animation of it
        run_script('Make_Animation.py')

    if simulation == "microstructure":
        # Run the main simulation script
        run_script('micro_structure_diffusion.py')

        # Run the script that makes you a nice animation of it
        run_script('Make_Animation_micro.py')
