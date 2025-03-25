import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from envs.custom_channel_env import NetworkEnvironment
from hybrid_trainer.metaheuristic_opt import run_metaheuristic
from hybrid_trainer.kpi_logger import KPI

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from envs.custom_channel_env import NetworkEnvironment
from hybrid_trainer.metaheuristic_opt import run_metaheuristic
from hybrid_trainer.kpi_logger import KPITracker
import time

def hybrid_training(config):
    """
    Runs hybrid training: 
    - Metaheuristic optimization for initialization 
    - RLlib MARL training with adaptive tuning 
    - Real-time KPI tracking
    """
    kpi_logger = KPITracker(log_file="logs/kpi_log.csv", live_plot=config["logging"]["live_plot"])
    
    if config["comparison_mode"]:
        algorithms_to_run = config["metaheuristic_algorithms"]
    else:
        algorithms_to_run = [config["metaheuristic"]]

    for meta_alg in algorithms_to_run:
        print(f"\n🔍 Running metaheuristic optimization: {meta_alg}...")
        meta_solution = run_metaheuristic(
            algorithm=meta_alg,
            num_bs=config["env_config"]["num_bs"],
            num_ue=config["env_config"]["num_ue"]
        )

        print(f"🚀 Starting RLlib MARL training with {config['marl_algorithm']}...")
        marl_config = PPOConfig().environment(NetworkEnvironment, env_config=config["env_config"]).resources(num_gpus=1).training()

        results = tune.run(
            "PPO",
            config=marl_config.to_dict(),
            stop={"training_iteration": config["training_iterations"]},
            checkpoint_at_end=True
        )

        # Log final KPIs
        final_metrics = kpi_logger.get_final_metrics()
        print(f"\n📊 Final KPI Results for {meta_alg}: {final_metrics}")

        # Adaptive tuning: Re-run metaheuristic if performance stagnates
        if config["adaptive_tuning"]["enabled"]:
            stagnation_threshold = config["adaptive_tuning"]["stagnation_threshold"]
            if final_metrics["reward"] < stagnation_threshold:
                print(f"⚠️ Performance stagnated ({final_metrics['reward']} < {stagnation_threshold}). Re-running metaheuristic...")
                time.sleep(2)
                continue  # Restart loop for this algorithm

    print("\n✅ Hybrid training complete!")
    kpi_logger.plot_final_comparison()

if __name__ == "__main__":
    ray.init()

    config = {
        "metaheuristic": "pfo",
        "marl_algorithm": "mappo",
        "comparison_mode": True,  # Run all metaheuristics for comparison
        "metaheuristic_algorithms": ["pfo", "aco", "pso"],
        "env_config": {
            "num_bs": 20,
            "num_ue": 200,
            "episode_length": 1000
        },
        "training_iterations": 1000,
        "logging": {
            "live_plot": True
        },
        "adaptive_tuning": {
            "enabled": True,
            "stagnation_threshold": -100  # Example reward threshold for re-tuning
        }
    }

    hybrid_training(config)


# def hybrid_training(config):
#     """
#     Runs hybrid training: Sequential Metaheuristic optimization + RLlib MARL (MAPPO) training.
#     Implements adaptive retraining if RL performance stagnates.
#     """
#     kpi_tracker = KPITracker(enabled=config["enable_logging"])
#     comparison_results = {}

#     for algo_name in config["metaheuristics"]:
#         print(f"Running metaheuristic optimization: {algo_name}...")
#         meta_solution = run_metaheuristic(
#             algorithm=algo_name,
#             num_bs=config["env_config"]["num_bs"],
#             num_ue=config["env_config"]["num_ue"]
#         )

#         print(f"Starting RLlib training with {algo_name} initialization...")
#         ppo_config = (
#             PPOConfig()
#             .environment(NetworkEnvironment, env_config=config["env_config"])
#             .resources(num_gpus=1)
#             .training(model={"custom_model_config": {"meta_solution": meta_solution}})
#         )

#         results = tune.run(
#             "PPO",
#             config=ppo_config.to_dict(),
#             stop={"training_iteration": config["training_iterations"]},
#             checkpoint_at_end=True
#         )

#         best_checkpoint = results.get_best_checkpoint()
#         avg_reward, avg_sinr, fairness, load_variance = kpi_tracker.evaluate_checkpoint(best_checkpoint)

#         comparison_results[algo_name] = {
#             "reward": avg_reward,
#             "sinr": avg_sinr,
#             "fairness": fairness,
#             "load_balance": load_variance
#         }

#         # Adaptive retraining if performance stagnates
#         if avg_reward < config["adaptive_threshold"]:
#             print(f"Performance stagnated for {algo_name}, re-running metaheuristic optimization...")
#             meta_solution = run_metaheuristic(
#                 algorithm=algo_name,
#                 num_bs=config["env_config"]["num_bs"],
#                 num_ue=config["env_config"]["num_ue"]
#             )
#             ppo_config.training(model={"custom_model_config": {"meta_solution": meta_solution}})
#             results = tune.run(
#                 "PPO",
#                 config=ppo_config.to_dict(),
#                 stop={"training_iteration": config["adaptive_retraining_iters"]},
#                 checkpoint_at_end=True
#             )

#     return comparison_results

# if __name__ == "__main__":
#     ray.init()

#     config = {
#         "metaheuristics": ["pfo", "aco", "pso"],  # Algorithms to compare
#         "env_config": {
#             "num_bs": 20,
#             "num_ue": 200,
#             "episode_length": 1000
#         },
#         "training_iterations": 1000,
#         "adaptive_threshold": -200,  # Retrain if reward stagnates below this value
#         "adaptive_retraining_iters": 500,
#         "enable_logging": True  # Toggle real-time logging ON/OFF
#         "real_time_plot": True,  # Toggle live graph updates
#     }

#     results = hybrid_training(config)
#     print("Comparison Results:", results)

