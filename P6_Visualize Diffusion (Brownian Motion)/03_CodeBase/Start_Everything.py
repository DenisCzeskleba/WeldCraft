"""
This is just to comfortably start the different simulations

"""

import sys
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run_script(script_name):
    script_path = SCRIPT_DIR / script_name
    try:
        subprocess.run([sys.executable, str(script_path)], cwd=SCRIPT_DIR, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {script_name}: {e}")


if __name__ == "__main__":

    # Run the main simulation script
    run_script('random motion.py')

    # Run the script that makes you a nice animation of it
    run_script('Make_Brownian_Animation.py')
