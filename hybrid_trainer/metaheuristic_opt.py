import sys
import os

# Add project root to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

import numpy as np
from typing import Dict, Any
from envs.custom_channel_env import NetworkEnvironment
from hybrid_trainer.kpi_logger import KPITracker
# from algorithms import aco, bat, cs, de, fa, ga, gwo, hs, ica, pfo, pso, sa, tabu, woa
# from algorithms.aco import ACO
# from algorithms.bat import BAT
# from algorithms.cs import CS
from algorithms.de import DEOptimization
# from algorithms.fa import FA
# from algorithms.ga import GA
# from algorithms.gwo import GWO
# from algorithms.hs import HS
# from algorithms.ica import ICA
# from algorithms.pfo import PFO
# from algorithms.pso import PSO
# from algorithms.sa import SA
# from algorithms.tabu import TABU
# from algorithms.woa import WOA

def run_metaheuristic(env: NetworkEnvironment, algorithm: str) -> dict:
    """
    Runs the selected metaheuristic algorithm and returns the optimized solution along with KPIs.
    
    :param algorithm: (str) Name of the metaheuristic algorithm to run.
    :param num_bs: (int) Number of base stations.
    :param num_ue: (int) Number of users.
    :return: Dict containing 'solution', 'SINR', 'fairness', 'load_balance', and 'handover_rate'.
    """
    algorithms = {
        # "aco": aco(num_ants=50, max_iter=100, decay=0.1, alpha=1.0, beta=2.0),
        # "bat": bat(population_size=50, num_iterations=100, frequency_range=(0, 1), loudness_decay=0.95),
        # "cs": cs(colony_size=30, iterations=50, pa=0.25),
        "de": DEOptimization(),
        # "fa": fa(population_size=30, iterations=50, beta0=1, gamma=1),
        # "ga": ga(population_size=30, generations=50, mutation_rate=0.1),
        # "gwo": gwo(swarm_size=30, iterations=50, a_initial=2.0, a_decay=0.04),
        # "hs": hs(memory_size=30, iterations=50, HMCR=0.9, PAR=0.3),
        # "ica": ica(population_size=30, imperialist_count=5, iterations=50),
        # "pfo": pfo(population_size=40, iterations=100, mutation_factor=0.2, jump_rate=0.2, follow_rate=0.3),
        # "pso": pso(swarm_size=30, iterations=50, c1=1, c2=1, w=0.5),
        # "sa": sa(iterations=100, initial_temp=100, cooling_rate=0.95),
        # "tabu": tabu(iterations=50, tabu_size=10),
        # "woa": woa(swarm_size=30, iterations=50),
    }

    if algorithm not in algorithms:
        raise ValueError(f"Invalid metaheuristic algorithm '{algorithm}'. Choose from: {list(algorithms.keys())}")

    print(f"Running {algorithm.upper()} for initial optimization...")
    solution = algorithms[algorithm].run(num_bs=num_bs, num_ue=num_ue)
    algo_instance = algorithms[algorithm]
    # Evaluate using environment's built-in method
    metrics = env.evaluate_detailed_solution(solution)
    
    # Log metrics using KPI tracker
    KPITracker().log_kpis(
        episode=0,
        reward=metrics["fitness"],
        sinr=metrics["average_sinr"],
        fairness=metrics["fairness"],
        load_variance=metrics["load_variance"]
    )
    
    return {
        "solution": solution,
        "metrics": metrics,
        "agents": get_agent_states(algo_instance)  # For visualization
        # "agents": get_agent_states(algorithm)  # For visualization
    }


def get_agent_states(algorithm_instance: Any) -> Dict:
    """Universal agent state extractor for all algorithms"""
    # Try common attribute patterns
    positions = _get_possible_attribute(
        algorithm_instance, 
        ["positions", "population", "swarm", "spatial_positions"]
    )
    
    fitness = _get_possible_attribute(
        algorithm_instance,
        ["fitness", "fitness_history", "best_fitnesses"]
    )
    
    # Special cases (e.g., ACO pheromones)
    trails = getattr(algorithm_instance, "pheromone_trails", None)
    
    return {
        "positions": _normalize_positions(positions),
        "fitness": _normalize_fitness(fitness),
        "trails": trails
    }

def _get_possible_attribute(obj, attr_names):
    """Safely get first matching attribute"""
    for name in attr_names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None  # Or default array

def _normalize_positions(positions):
    """Convert any position format to (x, y, fitness)"""
    if positions is None:
        return np.empty((0, 3))
    
    # Handle different formats
    if isinstance(positions, list):  # Common in population-based algorithms
        return np.array([[p[0], p[1], f] for p, f in zip(positions, fitness)])
    
    return positions  # Assume already formatted

def _normalize_fitness(fitness):
    """Ensure fitness is a numpy array"""
    return np.array(fitness) if fitness is not None else np.array([])

# def get_agent_states(algorithm_instance: Any) -> Dict:
#     """Universal agent state extractor based on algorithm properties"""
#     return {
#         "positions": getattr(algorithm_instance, 'positions', np.empty((0, 2))),
#         "fitness": getattr(algorithm_instance, 'fitness', np.array([])),
#         "velocity": getattr(algorithm_instance, 'velocity', None),  # Optional
#         "trails": getattr(algorithm_instance, 'pheromone_trails', None)  # For ACO
#     }  

    
