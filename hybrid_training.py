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
from hybrid_trainer.kpi_logger import KPITracker
import time

def hybrid_training(config):
    """
    Runs hybrid training: 
    - Metaheuristic optimization for initialization 
    - RLlib MARL training with adaptive tuning 
    - Real-time KPI tracking
    """
    
    # kpi_tracker = KPITracker(enabled=config["enable_logging"], real_time_plot=config["real_time_plot"])
    kpi_tracker = KPITracker(
    enabled=config["logging"]["enable_logging"],
    real_time_plot=config["logging"]["real_time_plot"]
                            )

    for algo_name in config["metaheuristic_algorithms"]:
        print(f"Training with {algo_name}...")
        
        for episode in range(config["training_iterations"]):
            avg_reward, avg_sinr, fairness, load_variance = kpi_tracker.evaluate_checkpoint("dummy_path")
            kpi_tracker.log_kpis(episode, avg_reward, avg_sinr, fairness, load_variance, algo_name)

    kpi_tracker.save_to_csv()
    kpi_tracker.plot_kpis(final=True)  # Generate final comparison graph

    print(f" Starting RLlib MARL training with {config['marl_algorithm']}...")
    marl_config = PPOConfig().environment(NetworkEnvironment, env_config=config["env_config"]).resources(num_gpus=1).training()

    results = tune.run(
        "PPO",
        config=marl_config.to_dict(),
        stop={"training_iteration": config["training_iterations"]},
        checkpoint_at_end=True
    )

    # Log final KPIs
    final_metrics = kpi_tracker.get_final_metrics()
    print(f"\n📊 Final KPI Results: {final_metrics}")

    # Adaptive tuning: Re-run metaheuristic if performance stagnates
    if config["adaptive_tuning"]["enabled"]:
        stagnation_threshold = config["adaptive_tuning"]["stagnation_threshold"]
        if final_metrics["reward"] < stagnation_threshold:
            print(f"⚠️ Performance stagnated ({final_metrics['reward']} < {stagnation_threshold}). Re-running metaheuristic...")
            time.sleep(2)
            hybrid_training(config)  # Restart training

    print("\n✅ Hybrid training complete!")
    kpi_tracker.plot_final_comparison()

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
            "enable_logging": True,
            "live_plot": True
        },
        "adaptive_tuning": {
            "enabled": True,
            "stagnation_threshold": -100  # Example reward threshold for re-tuning
        }
    }

    hybrid_training(config)
