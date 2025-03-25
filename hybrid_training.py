import sys
import os

# Add project root to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from envs.custom_channel_env import NetworkEnvironment
from hybrid_trainer.metaheuristic_opt import run_metaheuristic

def hybrid_training(config):
    """
    Runs hybrid training: Metaheuristic for initialization + RLlib MARL (MAPPO).
    """
    print(f"Running metaheuristic optimization ({config['metaheuristic']})...")
    meta_solution = run_metaheuristic(
        algorithm=config["metaheuristic"],
        num_bs=config["env_config"]["num_bs"],
        num_ue=config["env_config"]["num_ue"]
    )
    
    print("Starting RLlib MARL training (MAPPO)...")
    
    ppo_config = PPOConfig()
    ppo_config = (
        ppo_config.environment(NetworkEnvironment, env_config=config["env_config"])
        .resources(num_gpus=1)
        .training()
    )
    
    results = tune.run(
        "PPO",
        config=ppo_config.to_dict(),
        stop={"training_iteration": config["training_iterations"]},
        checkpoint_at_end=True
    )
    
    return results


if __name__ == "__main__":
    ray.init()
    
    config = {
        "metaheuristic": "pfo",  # Options: pfo, aco, pso, etc.
        "env_config": {
            "num_bs": 20,
            "num_ue": 200,
            "episode_length": 1000
        },
        "training_iterations": 1000
    }
    
    results = hybrid_training(config)
    print("Best checkpoint:", results.get_best_checkpoint())
