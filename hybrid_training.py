import sys
import os

# Configure project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

import ray
import time
import numpy as np
import pandas as pd
from ray import tune
# from ray.tune.trial import Trial
from typing import Dict
from IPython.display import display
from ray.rllib.algorithms.ppo import PPOConfig
from analysis.comparison import MetricAnimator 
from envs.custom_channel_env import NetworkEnvironment
from hybrid_trainer.metaheuristic_opt import run_metaheuristic
from hybrid_trainer.kpi_logger import KPITracker
from hybrid_trainer.live_dashboard import LiveDashboard

class HybridTraining:
    def __init__(self, config: Dict):
        self.config = config
        self.env = NetworkEnvironment(config["env_config"])
        self.kpi_logger = KPITracker(enabled=config["logging"]["enabled"])
        self.current_epoch = 0  # Track hybrid training epochs
        self.metaheuristic_runs = 0
        
        self.max_metaheuristic_runs = 1  
        self.dashboard = LiveDashboard(
            network_bounds=(0, 100)
            # algorithm_colors=self._init_algorithm_colors(),
            # update_interval=config["visualization"]["update_interval_ms"]
        )
        # display.display(self.dashboard.fig)
        
        ray.init(**config["ray_resources"])
        # Create log directory if needed
        if config["logging"]["enabled"]:
            os.makedirs(config["logging"]["log_dir"], exist_ok=True)
            
            
    def _init_algorithm_colors(self) -> Dict:
        """Map algorithms to visualization colors"""
        return {
            "pfo": "#FF6B6B",  # Coral
            "aco": "#4ECDC4",   # Teal
            "pso": "#45B7D1", # Sky Blue
            "de": "#FFA500",    # Orange 
            "marl": "#9B59B6"    # Purple
        }

    def _execute_metaheuristic_phase(self, algorithm: str) -> Dict:
        """Run a single metaheuristic optimization"""
        print(f"\n Initializing {algorithm.upper()} optimization...")
        # from IPython import display
        # Run optimization with visualization callback
       # Create a closure to capture algo state
        def de_visualize_callback(de_data: Dict):
            # display.clear_output(wait=True)  # Clear previous dashboard
            with self.dashboard.fig.batch_update():
                self.dashboard.update(
                        phase="metaheuristic",
                        data={
                            "env_state": self.env.get_current_state(),  # Pass env_state here
                            "metrics": {
                                "algorithm": algorithm,
                                "positions": de_data["positions"],
                                "fitness": de_data["fitness"]
                            }
                        }
                    )           
            # # Force Colab DOM update
            display(self.dashboard.fig)
            time.sleep(0.1)
        # solution = run_metaheuristic(self.env, algorithm)
        # Pass the visualization hook to the metaheuristic
        solution = run_metaheuristic(
            self.env,
            algorithm,
            self.current_epoch,
            kpi_logger=self.kpi_logger,
            visualize_callback= de_visualize_callback  # Proper data flow
        )
        
        print("Final KPI History after metaheuristic phase:")
        # print(self.kpi_logger.history)
        # Log and visualize results
        self.kpi_logger.log_algorithm_performance(algorithm=algorithm,metrics=solution["metrics"])
        # self.dashboard.update_algorithm_metrics(algorithm=algorithm,metrics=solution["metrics"] )
        
        return solution

    def _execute_marl_phase(self, initial_policy: Dict = None):
        """Execute MARL training phase"""
        # print(f"\n Starting {self.config['marl_algorithm'].upper()} training...")
        print(f"\n Starting {self.config.get('marl_algorithm', 'PPO').upper()} training...")
        # Configure environment with current network state
        env_config = {
            **self.config["env_config"],
            "current_state": self.env.get_current_state(),  # Capture initial state
            "environment_instance": self.env  # Pass through env_config instead
        }
        
        # marl_config = (PPOConfig()
        #     .environment(NetworkEnvironment, env_config=self.config["env_config"])
        #     .training(model={"custom_model": initial_policy} if initial_policy else {})
        #     .resources(num_gpus=self.config["marl_training"]["num_gpus"]))
        
        marl_config = (PPOConfig()
        .environment(
            NetworkEnvironment,
            env_config=env_config,            
        )
        .training(
            model={"custom_model": initial_policy} if initial_policy else {},
            num_sgd_iter=5,# num_epochs=5, 
            train_batch_size=4000
        )
        .resources(
            num_gpus=self.config["marl_training"]["num_gpus"],            
        )
        .env_runners(  # Add this section for CPU resources ✅
        num_cpus_per_env_runner=2  # Updated from num_cpus_per_worker
            )
        )
        
        analysis = tune.run(
            "PPO",
            config=marl_config.to_dict(),
            stop={"training_iteration": self.config["marl_steps_per_phase"]},
            checkpoint_at_end=True,
            callbacks=[self._create_marl_callback()]
        )
        
        return analysis

    def _create_marl_callback(self):
        """Generate callback for real-time MARL visualization"""
        class MarlVisualizationCallback(tune.Callback):
            def __init__(self, orchestrator):
                self.orchestrator = orchestrator
                self.last_env_state = None
                
            # def on_step_end(self, iteration, trials, trial, result, **info ):# **kwargs
            #     # Guard against empty trials or missing results
            #     if not trials or not trials[0].last_result:
            #         print("Missing results for trial!")
            #         return
            #     # Capture environment state from worker
            #     self.last_env_state = result.get("custom_metrics", {}).get("env_state")
                
            #     # Extract metrics
            #     metrics = {
            #         "episode_reward_mean": result.get("episode_reward_mean", 0),
            #         "average_sinr": result.get("custom_metrics", {}).get("sinr_mean", 0),
            #         "fairness": result.get("custom_metrics", {}).get("fairness_index", 0),
            #         "load_variance": result.get("custom_metrics", {}).get("load_variance", 0),
            #         "policy_entropy": result.get("policy_entropy", 0)
            #     }
                
               
            #     print(f"MARL Metrics at iteration {iteration} : {metrics}: ")
            #     self.orchestrator.kpi_logger.log_metrics(
            #         episode=iteration,
            #         phase="marl",
            #         algorithm= "PPO",
            #         metrics=metrics
            #     )
            def on_step_end(self, *, trial,result, **kwargs):
                """Fixed signature with required parameters"""
                # Extract metrics from result dict
                metrics = {
                    "episode_reward_mean": result.get("episode_reward_mean", 0),
                    "average_sinr": result.get("custom_metrics", {}).get("sinr_mean", 0),
                    "fairness": result.get("custom_metrics", {}).get("fairness_index", 0),
                    "load_variance": result.get("custom_metrics", {}).get("load_variance", 0),
                    "policy_entropy": result.get("policy_entropy", 0)
                }
                # Add formatted print with iteration number
                print(f"[Trial {trial.id}] Iter {result['training_iteration']} | "
                  f"Reward: {metrics['episode_reward_mean']:.1f} | "
                  f"SINR: {metrics['average_sinr']:.1f} dB")
                # print(f"MARL Metrics at iteration {result['training_iteration']}: {metrics}")
                # Log metrics through orchestrator
                self.orchestrator.kpi_logger.log_metrics(
                    episode=result["training_iteration"],
                    phase="marl",
                    algorithm="PPO",
                    metrics=metrics
                )
                
                #  Pull data from consolidated history
                
                # self.orchestrator.dashboard.update(
                #     phase="marl",
                #     data={
                #         "env_state": self.orchestrator.env.get_current_state(),  # Associations/users/BS
                #         # "metrics" : recent_metrics
                #         "metrics": {
                #         "episode_rewards_mean": metrics["episode_reward_mean"],
                #         "policy_entropy": metrics["policy_entropy"],
                #         "average_sinr": metrics["average_sinr"]
                #         }
                #     }
                #     )
                
                
                
                    
        return MarlVisualizationCallback(self)
    
    # def _create_marl_callback(self):
    #     """Generate callback for real-time MARL visualization"""
    #     class MarlVisualizationCallback(tune.Callback):
    #         def __init__(self, orchestrator):
    #             self.orchestrator = orchestrator
                
    #         def on_step_end(self, iteration, trials, **kwargs):
    #             # ✅ Log MARL metrics to KPI tracker
    #             self.orchestrator.kpi_logger.log_kpis(
    #                 episode=iteration,
    #                 reward=trials[0].last_result["episode_reward_mean"],
    #                 average_sinr=trials[0].last_result["custom_metrics"]["sinr_mean"],
    #                 fairness=trials[0].last_result["custom_metrics"]["fairness"],
    #                 load_variance=trials[0].last_result["custom_metrics"]["load_variance"]
    #             )
    #             metrics = self.orchestrator.kpi_logger.get_recent_metrics()
    #             self.orchestrator.dashboard.update(
    #                 env_state=self.orchestrator.env.get_current_state(),
    #                 metrics=metrics,
    #                 phase="marl"
    #             )
                
                
    #     return MarlVisualizationCallback(self)

    def _adaptive_retuning_required(self) -> bool:
        """Check if metaheuristic retuning is needed"""
        metrics = self.kpi_logger.get_recent_metrics(
            window_size=self.config["adaptive_tuning"]["stagnation_window"]
        )
        return (np.mean(metrics["reward"]) < 
                self.config["adaptive_tuning"]["stagnation_threshold"])

    def _compare_algorithms(self) -> Dict:
        """Run and compare multiple metaheuristics"""
        algorithm_results = {}
        
        for algo in self.config["metaheuristic_algorithms"]:
            self.env.reset()
            algorithm_results[algo] = self._execute_metaheuristic_phase(algo)
            time.sleep(1)  # Pause for visualization clarity
        # Create animator from logged history
        # List the metrics you want to animate
        metrics = ['fitness', 'average_sinr', 'fairness']

        # Loop over each metric to create, show, and save its animation separately
        for metric in metrics:
            # Create a MetricAnimator instance for the current metric
            animator = MetricAnimator(
                df=self.kpi_logger.history,
                metrics=[metric],  # Only process one metric at a time
                fps=8  # Lower FPS for slower progression
            )
            animator.animate()           # Build the animation for this metric
            animator.show()              # Render it inline in Jupyter/Colab
            animator.save_videos("results/separated_metrics")
        # animator = MetricAnimator(
        #     df=self.kpi_logger.history,
        #     metrics=['fitness'],#, ', 'average_sinr','fairness'
        #     fps=8  # Lower FPS for slower progression
        # )
        # animator.animate()
        # # For Jupyter
        # # Save to separate files
        # animator.show()
        # animator.save_videos("results/separated_metrics")
        

        # For video export
        # animator.save_videos("results/training_progression.mp4")  
        # Save to separate files
        
        # self.dashboard.display_comparison_matrix(algorithm_results)
        return algorithm_results

    def run(self):
        # """Main training orchestration"""
        try:
            if self.config["comparison_mode"]:
                algorithm_results = self._compare_algorithms()
                best_algorithm = max(
                    algorithm_results,
                    key=lambda x: algorithm_results[x]["metrics"]["fitness"]
                )
                print(f"\n Best algorithm selected: {best_algorithm.upper()}")
                initial_solution = algorithm_results[best_algorithm]
            else:
                # initial_solution = self._execute_metaheuristic_phase(
                #     self.config["metaheuristic"]
                # )
                # Run metaheuristic phase only if not already done
                if self.metaheuristic_runs < self.max_metaheuristic_runs:
                    initial_solution = self._execute_metaheuristic_phase(
                        self.config["metaheuristic"]
                    )
                    self.metaheuristic_runs += 1

            # Hybrid training loop
            current_phase = "marl"
            print(f"\n Current phase: {current_phase}")
            
            for epoch in range(1, self.config["max_epochs"] + 1):
                self.current_epoch = epoch
                if current_phase == "metaheuristic":
                    initial_solution = self._execute_metaheuristic_phase(
                        self.config["metaheuristic"]
                    )
                    current_phase = "marl"
                
                # Execute MARL phase
                analysis = self._execute_marl_phase(
                    initial_policy=initial_solution.get("policy")
                )
                
                # Log hybrid performance
                self.kpi_logger.log_epoch(
                    epoch=epoch,
                    marl_metrics=analysis.stats(),
                    metaheuristic_metrics=initial_solution["metrics"]
                )
                
                # Adaptive phase switching
                if (self.config["adaptive_tuning"]["enabled"] and 
                    self._adaptive_retuning_required()):
                    print("\n Performance stagnation detected - triggering retuning")
                    current_phase = "metaheuristic"
                
                # Save system state
                if epoch % self.config["checkpoint_interval"] == 0:
                    self.env.save_checkpoint(
                        f"{self.config['checkpoint_dir']}/epoch_{epoch}.pkl"
                    )

        finally:
           # self.dashboard.finalize_visualizations()
            self.kpi_logger.generate_final_reports()
            ray.shutdown()


if __name__ == "__main__":
    
    config = {
        # Core configuration
        "metaheuristic": "bat",
        "comparison_mode": True,
        "metaheuristic_algorithms": ["aco","bat", "cs", "de", "fa", "ga", "gwo", "hs", "ica", "pfo", "pso", "sa", "tabu", "woa"], #
        "marl_algorithm": "PPO", 
        
        # Environment parameters
        "env_config": {
            "num_bs": 20,
            "num_ue": 200,
            "episode_length": 1000,
            "log_kpis": True
        },
        
                        
        # Training parameters
        "max_epochs": 50,
        "marl_steps_per_phase": 200,
        "checkpoint_interval": 10,
        "checkpoint_dir": "results/checkpoints",
        
        # Resource management
        "ray_resources": {
            "num_cpus": 8,
            "num_gpus": 1,
            },
        "marl_training": {
        "num_gpus": 0.5  #  GPUs allocated to MARL
        },
        # Visualization parameters
        "visualization": {
            # "update_interval_ms": 500,
            "3d_resolution": 20,
            "heatmap_bins": 15
        },
        
        # Adaptive control
        "adaptive_tuning": {
            "enabled": True,
            "stagnation_threshold": -50,
            "stagnation_window": 10
        },
        
        # Logging
        "logging": {
            "enabled": True,
            "log_dir": "results/logs"
        }
    }

    orchestrator = HybridTraining(config)
    orchestrator.run()
