# ============================================================
# RUN SIMULATION — runner script
# ============================================================

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from simulation.employee_simulator import simulate_workforce

if __name__ == "__main__":
    simulate_workforce(n=20, scenario="random")
