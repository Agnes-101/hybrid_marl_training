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
from algorithms.aco import ACO
from algorithms.bat import BatOptimization
from algorithms.cs import CSOptimization
from algorithms.de import DEOptimization
from algorithms.fa import FireflyOptimization
from algorithms.ga import GAOptimization
from algorithms.gwo import GWOOptimization
from algorithms.hs import HarmonySearchOptimization
from algorithms.ica import ICAOptimization
from algorithms.pfo import PolarFoxOptimization
from algorithms.pso import PSOOptimization
from algorithms.sa import SAOptimization
from algorithms.tabu import TabuSearchOptimization
from algorithms.woa import WOAOptimization

def run_metaheuristic(env: NetworkEnvironment, algorithm: str, epoch: int, kpi_logger: KPITracker,visualize_callback=None) -> dict:
    """
    Runs the selected metaheuristic algorithm and returns the optimized solution along with KPIs.
    
    :param algorithm: (str) Name of the metaheuristic algorithm to run.
    :param num_bs: (int) Number of base stations.
    :param num_ue: (int) Number of users.
    :return: Dict containing 'solution', 'SINR', 'fairness', 'load_balance', and 'handover_rate'.
    """
    algorithms = {
        "aco": ACO,
        "bat": BatOptimization,
        "cs": CSOptimization,
        "de": DEOptimization,
        "fa": FireflyOptimization,
        "ga": GAOptimization,
        "gwo": GWOOptimization,
        "hs": HarmonySearchOptimization,
        "ica": ICAOptimization,
        "pfo": PolarFoxOptimization,
        "pso": PSOOptimization,
        "sa": SAOptimization,
        "tabu": TabuSearchOptimization,
        "woa": WOAOptimization,
    }

    if algorithm not in algorithms:
        raise ValueError(f"Invalid metaheuristic algorithm '{algorithm}'. Choose from: {list(algorithms.keys())}")

    print(f"Running {algorithm.upper()} for initial optimization...")
    # Modified line: pass the environment to the run method
    # Instantiate with required parameters
    algo_class = algorithms[algorithm]
    algo_instance = algo_class(env=env,kpi_logger=kpi_logger)  # ✅ Pass logger

    print(f"\n Algorithm Instance, {algo_instance}")
    # solution = algo_instance.run(env)
    # Pass BOTH logger and callback to Algorithm's run()
    solution_data = algo_instance.run(
        # env=env,
        visualize_callback=visualize_callback,  # Critical
        kpi_logger=kpi_logger
    )
    solution= solution_data.get('solution')
    print(f"\n Algorithm Solution, {solution}")
    
    # Extract solution from the result
    # solution = solution_result["solution"]
    # solution = algorithms[algorithm].run(num_bs=num_bs, num_ue=num_ue).run(env)
    
    
    # # Evaluate using environment's built-in method
    metrics = env.evaluate_detailed_solution(solution)
    print(f"\n Metrics, {metrics}")
    
    
    # # Log metrics using KPI tracker
    # kpi_logger.log_kpis(
    #     episode=epoch,
    #     reward=metrics["fitness"],
    #     sinr=metrics["average_sinr"],
    #     fairness=metrics["fairness"],
    #     load_variance=metrics["load_variance"]
    # )
    
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

    
