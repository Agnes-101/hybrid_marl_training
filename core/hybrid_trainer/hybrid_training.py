#hybrid_training.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))# ".."
sys.path.insert(0, project_root)if project_root not in sys.path else None
print(f"Verified Project Root: {project_root}")  # Should NOT be "/"


# ENVIRONMENT REGISTRATION MUST be outside class definition
from ray.tune.registry import register_env
from core.envs.custom_channel_env import NetworkEnvironment

def env_creator(env_config):
    return NetworkEnvironment(env_config)

register_env("NetworkEnv", env_creator)


import ray
import time
import torch
import numpy as np
import pandas as pd
from ray import tune
from ray.rllib.models import ModelCatalog
# from ray.tune.trial import Trial
from typing import Dict
from ray.rllib.algorithms.ppo import PPOConfig
from core.analysis.comparison import MetricAnimator
from core.hybrid_trainer.metaheuristic_opt import run_metaheuristic
from core.hybrid_trainer.kpi_logger import KPITracker
from core.hybrid_trainer.live_dashboard import LiveDashboard
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import gymnasium as gym
from collections import OrderedDict


class MetaPolicy(TorchModelV2, nn.Module):
    # def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
    #     # super().__init__(obs_space, action_space, num_outputs, model_config, name)
    #     nn.Module.__init__(self)  # Initialize nn.Module first
    #     TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
    #     print("Model Config:", model_config["custom_model_config"])  # Debug line
    #     # Extract metaheuristic solution from config
    #     self.initial_weights = model_config["custom_model_config"].get("initial_weights", [])
    #     self.num_bs = model_config["custom_model_config"]["num_bs"]
    #     self.num_ue = model_config["custom_model_config"]["num_ue"]
        
    #     # Define neural network layers
    #     self.fc = torch.nn.Linear(obs_space.shape[0], self.num_bs)
        
    #     # Initialize weights using metaheuristic solution
    #     if self.initial_weights:
    #         self._apply_initial_weights()
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        print("Starting MetaPolicy initialization")
        
        # Initialize in correct order
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        
        # Print detailed info about inputs
        print(f"obs_space: {obs_space} (type: {type(obs_space)})")
        print(f"action_space: {action_space} (type: {type(action_space)})")
        print(f"num_outputs: {num_outputs}")
        print(f"model_config keys: {list(model_config.keys())}")
        
        # Get custom config safely
        custom_config = model_config.get("custom_model_config", {})
        print(f"custom_model_config: {custom_config}")
        
        # Extract parameters
        self.initial_weights = custom_config.get("initial_weights", [])
        self.num_bs = custom_config.get("num_bs", 5)
        self.num_ue = custom_config.get("num_ue", 20)
        
        print(f"Extracted values: initial_weights={len(self.initial_weights)} items, num_bs={self.num_bs}, num_ue={self.num_ue}")
        
        # Determine input size from observation space
        try:
            if isinstance(obs_space, gym.spaces.Dict):
                print(f"Dict observation space with keys: {list(obs_space.spaces.keys())}")
                input_size = sum(np.prod(space.shape) for space in obs_space.spaces.values())
            else:
                input_size = np.prod(obs_space.shape)
            print(f"Calculated input_size: {input_size}")
        except Exception as e:
            print(f"Error determining input size: {e}")
            # Fallback
            input_size = 20  # Default value, adjust based on your actual observation size
            print(f"Using fallback input_size: {input_size}")
        
        # Network layers
        try:
            self.fc = torch.nn.Linear(input_size, self.num_bs)
            print("Successfully created neural network layers")
        except Exception as e:
            print(f"Error creating network layers: {e}")
            
        # Initialize weights
        if self.initial_weights:
            try:
                self._apply_initial_weights()
                print("Successfully applied initial weights")
            except Exception as e:
                print(f"Error applying initial weights: {e}")
                
        print("Completed MetaPolicy initialization")
        
    # def _apply_initial_weights(self):
    #     """Bias policy to favor initial solution's BS choices"""
    #     # Convert initial solution to one-hot encoded tensor
    #     # Example: initial_weights = [1, 3, 0, ...] → UE 0 → BS 1, UE 1 → BS 3, etc.
    #     print(f"Initial solution shape: {len(self.initial_weights)} UEs")  
        
    #     one_hot_weights = torch.eye(self.num_bs)[self.initial_weights]  # Shape: [num_ue, num_bs]
    #     print(f"One-hot weights shape: {one_hot_weights.shape} (UEs x BSs)")
    #     print("Sample weights for UE 0:", one_hot_weights[0])  # Should be one-hot
        
    #     # Set linear layer weights to favor initial solution
    #     with torch.no_grad():
    #         self.fc.weight.data = one_hot_weights.float()
    #         self.fc.bias.data.zero_()
    def _apply_initial_weights(self):
        """Bias policy to favor initial solution's BS choices"""
        # Create one-hot encoded weights
        one_hot_weights = torch.eye(self.num_bs)[self.initial_weights]  # Shape: [num_ue, num_bs]
        
        # Linear layer weight shape needs to be [output_size, input_size]
        # But we need to map from input_size (220) to num_bs (5)
        with torch.no_grad():
            # Reshape the weights to properly initialize the linear layer
            # Initialize most weights to near-zero
            self.fc.weight.data.fill_(0.01)
            
            # For each UE, strengthen connections from its section of the input
            # to the BS it was assigned in the initial solution
            input_per_ue = self.fc.weight.data.shape[1] // self.num_ue
            
            for ue_idx, bs_idx in enumerate(self.initial_weights):
                # Each UE gets a section of the input
                start_idx = ue_idx * input_per_ue
                end_idx = (ue_idx + 1) * input_per_ue
                
                # Set those weights higher for the assigned BS
                self.fc.weight.data[bs_idx, start_idx:end_idx] = 1.0
            
            # Initialize bias
            self.fc.bias.data.zero_()
            
            print(f"Weight matrix shape: {self.fc.weight.data.shape}")
    
    def forward(self, input_dict, state, seq_lens):
        # Get observation
        obs = input_dict["obs"]
        
        # Handle different observation types
        if isinstance(obs, dict) or isinstance(obs, OrderedDict):
            # For dictionary observations
            print(f"Dict observation with keys: {list(obs.keys())}")
            # Convert dict values to tensor
            x = torch.cat([v.float() for v in obs.values()], dim=-1)
        else:
            # For tensor observations
            print(f"Tensor observation with shape: {obs.shape}")
            x = obs
            
        # Forward pass through network
        logits = self.fc(x)
        return logits, state
    # def forward(self, input_dict, state, seq_lens):
    #     # Handle input that could be a dictionary or tensor
    #     print(f"Forward called with input_dict keys: {list(input_dict.keys())}")
    #     print(f"Observation shape: {input_dict['obs'].shape}")  # Debug line
        
    #     x = input_dict["obs"]
        
    #     # Debug the input type
    #     print(f"Input type: {type(x)}")
        
    #     # # If observation is a dictionary, extract the relevant part
    #     # if isinstance(x, dict) or isinstance(x, OrderedDict):
    #     #     # Choose the appropriate part of the observation
    #     #     # You may need to adjust this based on your actual observation structure
    #     #     x = next(iter(x.values()))  # Get first value in dict
    #     #     print(f"Using first value from dict with shape: {x.shape}")
    #     # Get observation
    #     obs = input_dict["obs"]
        
    #     # Handle dict observation
    #     if isinstance(obs, dict) or isinstance(obs, OrderedDict):
    #         # Convert dict to tensor
    #         obs_values = list(obs.values())
    #         if obs_values:
    #             x = torch.cat([v.float() for v in obs_values if hasattr(v, 'float')], dim=-1)
    #         else:
    #             raise ValueError("Empty observation dictionary")
    #     else:
    #         # Already a tensor
    #         x = obs
    #     # Now process the tensor
    #     logits = self.fc(x)
    #     return logits, state
# class MetaPolicy(TorchModelV2):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         super().__init__(obs_space, action_space, num_outputs, model_config, name)
        
#         # Initialize layers using metaheuristic weights
#         initial_weights = model_config["custom_model_config"].get("initial_weights", [])
#         if initial_weights:
#             self.load_metaheuristic_weights(initial_weights)

#     def load_metaheuristic_weights(self, weights):
#         """Map metaheuristic solution to policy weights"""
#         # Example: Create bias toward GA's BS choices
#         with torch.no_grad():
#             for param in self.parameters():
#                 if param.dim() == 2:  # Weight matrices
#                     torch.nn.init.eye_(param)
#                 elif param.dim() == 1:  # Biases
#                     param.data += torch.tensor(weights, dtype=torch.float32)
    
class MetaPolicyRLModule(RLModule):
    def __init__(self, observation_space, action_space, model_config):
        super().__init__(observation_space, action_space, model_config)
        self.torch_model = MetaPolicy(
            observation_space,
            action_space,
            model_config["custom_model_config"]
        )
# After MetaPolicy definition
ModelCatalog.register_custom_model("meta_policy", MetaPolicy)

# RAY_DEDUP_LOGS=0
class HybridTraining:
    def __init__(self, config: Dict):
        # Initialize Ray AFTER path modification        
        ray.init(
            runtime_env={
                "env_vars": {"PYTHONPATH": project_root},
                "working_dir": project_root
            },
            **config["ray_resources"]
        )
        # ray.init(
        #     runtime_env={
        #         "env_vars": {"PYTHONPATH":f"{project_root}:{os.environ.get('PYTHONPATH', '')}"},
        #                                 # {"PYTHONPATH": hybrid_marl_dir},
        #         "working_dir":project_root,# hybrid_marl_dir,                 
        #         # Block problematic paths
        #         "includes": [  # Explicitly include critical paths
        #             "envs/",
        #             "hybrid_trainer/",
        #             "config/*.yaml"
        #         ],
        #         "excludes": [
        #             "**/sys/**", 
        #             "**/results/**",  # Exclude large outputs
        #             "**/notebooks/**",  # Exclude Colab/IPYNB files
        #             "*.pyc",
        #             "__pycache__/",
        #             ".*"
        #         ]
        #     },
        #     **config["ray_resources"]
        # )
        @ray.remote
        def verify_package():
            try:
                from core.envs.custom_channel_env import NetworkEnvironment
                from core.hybrid_trainer.hybrid_training import HybridTraining
                return True
            except ImportError as e:
                print(f"Import failed: {e}")
                return False
        assert ray.get(verify_package.remote()), "Package verification failed!"
        # ✅ Define observation/action spaces from the environment
        
        self.config = config
        self.env = NetworkEnvironment(config["env_config"])
        self.obs_space = self.env.observation_space
        self.act_space = self.env.action_space
        self.kpi_logger = KPITracker(enabled=config["logging"]["enabled"])
        self.current_epoch = 0  # Track hybrid training epochs
        self.metaheuristic_runs = 0
        
        self.max_metaheuristic_runs = 1  
        # self.dashboard = LiveDashboard(
        #     network_bounds=(0, 100)
        #     # algorithm_colors=self._init_algorithm_colors(),
        #     # update_interval=config["visualization"]["update_interval_ms"]
        # )
        # display.display(self.dashboard.fig)
        
        # ray.init(**config["ray_resources"])
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
            # display(self.dashboard.fig)
            time.sleep(0.1)
        # solution = run_metaheuristic(self.env, algorithm)
        # Pass the visualization hook to the metaheuristic
        solution = run_metaheuristic(
            self.env,
            algorithm,
            self.current_epoch,
            kpi_logger=self.kpi_logger,
            visualize_callback= None # Proper data flow
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
        # Extract initial solution safely
        initial_weights = []
        if initial_policy is not None:
            initial_weights = initial_policy.tolist()  # ✅ Correct extraction
            assert len(initial_weights) == self.config["env_config"]["num_ue"]
            
        env_config = {
            **self.config["env_config"]            
        }
        
        
        # MARL configuration
        marl_config = (
            PPOConfig()
            .environment(
                "NetworkEnv",
                env_config=env_config
            )
            .training(
                model={
                    "custom_model": "meta_policy",
                    "custom_model_config": {
                        "initial_weights": initial_weights,
                        "num_bs": self.config["env_config"]["num_bs"],
                        "num_ue": self.config["env_config"]["num_ue"]
                    }
                },
                gamma=0.99,
                lr=0.0001,
                kl_coeff=0.3,
                train_batch_size=1000
            )
            .multi_agent(
                policies={
                    f"ue_{i}": (None, self.obs_space, self.act_space, {})
                    for i in range(self.config["env_config"]["num_ue"])
                },
                # policy_mapping_fn=lambda agent_id: agent_id
                policy_mapping_fn=lambda agent_id: (
                    print(f"Mapping agent: {agent_id}") or agent_id  # Debug line
            )            
            )
        )
        # Before tune.run()
        # if initial_policy:
        #     # ModelCatalog.register_custom_model("meta_policy", type(initial_policy))
        #     ModelCatalog.register_custom_model("meta_policy", MetaPolicy)
            
        analysis = ray.tune.run(
            "PPO",
            config=marl_config.to_dict(),
            stop={"training_iteration": self.config["marl_steps_per_phase"]},
            checkpoint_at_end=True,
            callbacks=[self._create_marl_callback()]
        )
        print("Trial errors:", analysis.errors)
        return analysis
        # Add this after tune.run()
        

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
            def on_trial_result(self, *, trial,result, **kwargs):
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
                print(f"Initial Solution is : {initial_solution}")
            # Hybrid training loop
            current_phase = "marl"
            print(f"\n Current phase: {current_phase}")
            
            for epoch in range(1, self.config["max_epochs"] + 1):
                self.current_epoch = epoch
                if current_phase == "metaheuristic":
                    initial_solution = self._execute_metaheuristic_phase(
                        self.config["metaheuristic"]
                    )
                    print(f"Initial Solution is : {initial_solution}")
                    current_phase = "marl"
                
                # Execute MARL phase
                analysis = self._execute_marl_phase(
                    initial_policy=initial_solution.get("solution")
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
        "metaheuristic": "sa",
        "comparison_mode": False,
        "metaheuristic_algorithms": ["aco","bat", "cs", "de", "fa", "ga", "gwo", "hs", "ica", "pfo", "pso", "sa", "tabu", "woa"], #
        "marl_algorithm": "PPO", 
        
        # Environment parameters
        "env_config": {
            "num_bs": 5,
            "num_ue": 20,
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
            "num_gpus": 0,
            },
        "marl_training": {
        "num_gpus": 0  #  GPUs allocated to MARL
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
