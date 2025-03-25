import sys
import os

# Add project root to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

import numpy as np
from hybrid_trainer.kpi_logger import KPITracker
from algorithms import aco, bat, cs, de, fa, ga, gwo, hs, ica, pfo, pso, sa, tabu, woa

def run_metaheuristic(algorithm, num_bs, num_ue):
    """
    Runs the selected metaheuristic algorithm and returns the optimized solution along with KPIs.
    
    :param algorithm: (str) Name of the metaheuristic algorithm to run.
    :param num_bs: (int) Number of base stations.
    :param num_ue: (int) Number of users.
    :return: Dict containing 'solution', 'SINR', 'fairness', 'load_balance', and 'handover_rate'.
    """
    algorithms = {
        "aco": aco(num_ants=50, max_iter=100, decay=0.1, alpha=1.0, beta=2.0),
        "bat": bat(population_size=50, num_iterations=100, frequency_range=(0, 1), loudness_decay=0.95),
        "cs": cs(colony_size=30, iterations=50, pa=0.25),
        "de": de(population_size=30, iterations=50, F=0.8, CR=0.9),
        "fa": fa(population_size=30, iterations=50, beta0=1, gamma=1),
        "ga": ga(population_size=30, generations=50, mutation_rate=0.1),
        "gwo": gwo(swarm_size=30, iterations=50, a_initial=2.0, a_decay=0.04),
        "hs": hs(memory_size=30, iterations=50, HMCR=0.9, PAR=0.3),
        "ica": ica(population_size=30, imperialist_count=5, iterations=50),
        "pfo": pfo(population_size=40, iterations=100, mutation_factor=0.2, jump_rate=0.2, follow_rate=0.3),
        "pso": pso(swarm_size=30, iterations=50, c1=1, c2=1, w=0.5),
        "sa": sa(iterations=100, initial_temp=100, cooling_rate=0.95),
        "tabu": tabu(iterations=50, tabu_size=10),
        "woa": woa(swarm_size=30, iterations=50),
    }

    if algorithm not in algorithms:
        raise ValueError(f"Invalid metaheuristic algorithm '{algorithm}'. Choose from: {list(algorithms.keys())}")

    print(f"Running {algorithm.upper()} for initial optimization...")
    optimized_solution = algorithms[algorithm].run(num_bs=num_bs, num_ue=num_ue)

    # Compute KPIs based on the optimized solution
    sinr = np.random.uniform(10, 40)  # Placeholder SINR computation
    fairness = np.random.uniform(0.5, 1)  # Placeholder Fairness Index
    load_balance = np.random.uniform(0.2, 1)  # Placeholder Load Balancing Score
    handover_rate = np.random.uniform(0, 0.2)  # Placeholder Handover Rate

    # Log KPIs for tracking
    logger = KPITracker(enable_tracking=True)
    logger.log_kpis(episode=0, sinr=sinr, fairness=fairness, load_balance=load_balance, handover_rate=handover_rate)

    return {
        "solution": optimized_solution,
        "SINR": sinr,
        "fairness": fairness,
        "load_balance": load_balance,
        "handover_rate": handover_rate
    }

