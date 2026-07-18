"""
Convenience starter for the Brownian motion simulation and animation.
"""

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run_script(script_name):
    script_path = SCRIPT_DIR / script_name
    try:
        subprocess.run([sys.executable, str(script_path)], cwd=SCRIPT_DIR, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {script_name}: {e}")
        return False


if __name__ == "__main__":
    if run_script("b1_Random_Motion.py"):
        run_script("c1_Brown_Make_Animation.py")
