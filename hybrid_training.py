import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.maddpg import MADDPGConfig
from ray.rllib.algorithms.qmix import QMixConfig
from envs.custom_channel_env import NetworkEnvironment
from hybrid_trainer.metaheuristic_opt import run_metaheuristic
import numpy as np

def hybrid_training(config):
    """
    Runs hybrid training for multiple metaheuristic algorithms in batch mode.
    """
    print("Running batch metaheuristic optimization...")

    meta_results = {}
    for algo in config["metaheuristics"]:
        print(f"Running {algo.upper()} for initial optimization...")
        meta_results[algo] = run_metaheuristic(
            algorithm=algo,
            num_bs=config["env_config"]["num_bs"],
            num_ue=config["env_config"]["num_ue"]
        )

    # Compare Metaheuristic Performance
    print("\n=== Metaheuristic Algorithm Comparison ===")
    best_algo = None
    best_fairness = -np.inf  # Higher is better

    for algo, results in meta_results.items():
        print(f"\nAlgorithm: {algo.upper()}")
        print(f"  SINR: {results['SINR']:.2f} dB")
        print(f"  Fairness: {results['fairness']:.4f}")
        print(f"  Load Balance: {results['load_balance']:.4f}")
        print(f"  Handover Rate: {results['handover_rate']:.4f}")

        if results["fairness"] > best_fairness:  # Choosing the best based on fairness
            best_fairness = results["fairness"]
            best_algo = algo

    print(f"\n✅ Best Metaheuristic Algorithm: {best_algo.upper()} (Fairness: {best_fairness:.4f})")

    # Use the best algorithm’s solution for MARL training
    meta_solution = meta_results[best_algo]["solution"]

    print(f"Starting RLlib MARL training ({config['marl_algorithm']})...")

    marl_configs = {
        "mappo": PPOConfig,
        "maddpg": MADDPGConfig,
        "qmix": QMixConfig
    }

    if config["marl_algorithm"] not in marl_configs:
        raise ValueError("Invalid MARL algorithm. Choose from: mappo, maddpg, qmix")

    marl_config = marl_configs[config["marl_algorithm"]]()
    marl_config = (
        marl_config.environment(NetworkEnvironment, env_config=config["env_config"])
        .resources(num_gpus=1)
        .training()
    )

    results = tune.run(
        config["marl_algorithm"].upper(),
        config=marl_config.to_dict(),
        stop={"training_iteration": config["training_iterations"]},
        checkpoint_at_end=True
    )

    return results

if __name__ == "__main__":
    ray.init()

    config = {
        "metaheuristics": ["pfo", "pso", "ga"],  # Multiple metaheuristic algorithms
        "marl_algorithm": "mappo",  # Options: mappo, maddpg, qmix
        "env_config": {
            "num_bs": 20,
            "num_ue": 200,
            "episode_length": 1000
        },
        "training_iterations": 1000
    }

    results = hybrid_training(config)
    print("Best checkpoint:", results.get_best_checkpoint())
