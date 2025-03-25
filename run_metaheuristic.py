import sys
import os

# Add project root to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

import numpy as np
from algorithms.aco import ACO
from algorithms.bat import BAT
from algorithms.cs import CS
from algorithms.de import DE
from algorithms.fa import FA
from algorithms.ga import GA
from algorithms.gwo import GWO
from algorithms.hs import HS
from algorithms.ica import ICA
from algorithms.pfo import PFO
from algorithms.pso import PSO
from algorithms.sa import SA
from algorithms.tabu import TABU
from algorithms.woa import WOA

# Dictionary mapping algorithm names to classes
META_ALGORITHMS = {
    "aco": ACO,
    "bat": BAT,
    "cs": CS,
    "de": DE,
    "fa": FA,
    "ga": GA,
    "gwo": GWO,
    "hs": HS,
    "ica": ICA,
    "pfo": PFO,
    "pso": PSO,
    "sa": SA,
    "tabu": TABU,
    "woa": WOA
}

def run_metaheuristic(algorithm, num_bs, num_ue):
    """
    Runs the selected metaheuristic optimization algorithm to initialize user association.
    
    Args:
        algorithm (str): Name of the metaheuristic algorithm (e.g., 'pfo', 'pso', 'ga').
        num_bs (int): Number of base stations.
        num_ue (int): Number of users.
    
    Returns:
        np.array: Optimized user-to-BS assignment solution.
    """
    
    if algorithm not in META_ALGORITHMS:
        raise ValueError(f"Invalid algorithm '{algorithm}'. Choose from: {list(META_ALGORITHMS.keys())}")
    
    print(f"Running {algorithm.upper()} optimization for user association...")
    
    # Initialize the selected metaheuristic
    meta_solver = META_ALGORITHMS[algorithm](
        population_size=30, iterations=50  # Default hyperparameters, can be tuned
    )
    
    # Define objective function: Maximizing SINR & fairness
    def objective_function(solution):
        """
        Objective function for metaheuristic optimization.
        Solution represents user-to-BS assignments.
        """
        bs_loads = np.zeros(num_bs)
        sinr_values = np.zeros(num_ue)

        for ue in range(num_ue):
            bs_id = int(solution[ue])  # Assigned base station
            bs_loads[bs_id] += 1  # Track BS load
            distance = np.random.uniform(10, 100)  # Approximate distance
            path_loss = 1 / (distance ** 2 + 1e-6)  # Simplified path loss model
            sinr_values[ue] = path_loss  # SINR estimation

        fairness = np.sum(bs_loads) ** 2 / (num_bs * np.sum(bs_loads ** 2) + 1e-6)
        return -(np.mean(sinr_values) + fairness)  # Minimize negative SINR + fairness

    # Run the optimization
    best_solution, best_fitness = meta_solver.optimize(objective_function, num_ue, bounds=(0, num_bs - 1))

    print(f"Metaheuristic {algorithm.upper()} completed. Best fitness: {best_fitness}")
    
    return np.round(best_solution).astype(int)  # Convert to discrete BS assignments
