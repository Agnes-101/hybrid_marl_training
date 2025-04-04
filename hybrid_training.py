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
from typing import Dict
from IPython import display
from ray.rllib.algorithms.ppo import PPOConfig
from envs.custom_channel_env import NetworkEnvironment
from hybrid_trainer.metaheuristic_opt import run_metaheuristic
from hybrid_trainer.kpi_logger import KPITracker
from hybrid_trainer.live_dashboard import LiveDashboard

class HybridTraining:
    def __init__(self, config: Dict):
        self.config = config
        self.env = NetworkEnvironment(**config["env_config"])
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
            # display.display(self.dashboard.fig)
            # time.sleep(0.1)
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

        
        marl_config = (PPOConfig()
            .environment(NetworkEnvironment, env_config=self.config["env_config"])
            .training(model={"custom_model": initial_policy} if initial_policy else {})
            .resources(num_gpus=self.config["marl_training"]["num_gpus"]))
        
        # marl_config = (PPOConfig()
        #     .environment(NetworkEnvironment, env_config=self.config["env_config"])
        #     .training(
        #         model={"custom_model": initial_policy} if initial_policy else {},
        #         num_cpus_per_worker=2  # ✅ Correct parameter location
        #     )
        #     .resources(
        #         num_gpus=self.config["marl_training"]["num_gpus"],
        #         num_learner_workers=1  # Optional but recommended
        #     ))
        
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
                
            def on_step_end(self, iteration, trials, **kwargs):
                # ✅ Unified logging for MARL metrics
                metrics = {
                    "fitness": trials[0].last_result["episode_reward_mean"],
                    "average_sinr": trials[0].last_result["custom_metrics"]["sinr_mean"],
                    "fairness": trials[0].last_result["custom_metrics"]["fairness"],
                    "load_variance": trials[0].last_result["custom_metrics"]["load_variance"]
                }
                
                self.orchestrator.kpi_logger.log_metrics(
                    episode=iteration,
                    phase="marl",
                    algorithm= "PPO",
                    metrics=metrics
                )
                
                #  Pull data from consolidated history
                recent_metrics = self.orchestrator.kpi_logger.get_recent_metrics()
                # self.orchestrator.dashboard.update(
                #     env_state=self.orchestrator.env.get_current_state(),
                #     metrics=recent_metrics,
                #     phase="marl"
                self.orchestrator.dashboard.update(
                    phase="marl",
                    data={
                        "env_state": self.orchestrator.env.get_current_state(),  # Associations/users/BS
                        "metrics" : recent_metrics
                        # "metrics": {                                # Episode rewards/entropy
                        #     "episode_rewards": marl_data["rewards"],
                        #     "policy_entropy": marl_data["entropy"]
                        # }
                    }
                )
                
                
                    
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
            
        self.dashboard.display_comparison_matrix(algorithm_results)
        return algorithm_results

    def run(self):
        # """Main training orchestration"""
        # try:
        #     if self.config["comparison_mode"]:
        #         algorithm_results = self._compare_algorithms()
        #         best_algorithm = max(
        #             algorithm_results,
        #             key=lambda x: algorithm_results[x]["metrics"]["fitness_score"]
        #         )
        #         print(f"\n Best algorithm selected: {best_algorithm.upper()}")
        #         initial_solution = algorithm_results[best_algorithm]
        #     else:
        #         # initial_solution = self._execute_metaheuristic_phase(
        #         #     self.config["metaheuristic"]
        #         # )
        #         # Run metaheuristic phase only if not already done
        #         if self.metaheuristic_runs < self.max_metaheuristic_runs:
        #             initial_solution = self._execute_metaheuristic_phase(
        #                 self.config["metaheuristic"]
        #             )
        #             self.metaheuristic_runs += 1

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
        "metaheuristic": "de",
        "comparison_mode": False,
        "metaheuristic_algorithms": ["pfo", "aco", "pso"],
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
