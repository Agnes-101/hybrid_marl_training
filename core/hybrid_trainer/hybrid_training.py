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
import logging
import torch
import numpy as np
import pandas as pd
from ray import tune
from ray.rllib.models import ModelCatalog
# from ray.tune.trial import Trial
from typing import Dict
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO
from core.analysis.comparison import MetricAnimator
from core.hybrid_trainer.metaheuristic_opt import run_metaheuristic
from core.hybrid_trainer.kpi_logger import KPITracker
from core.hybrid_trainer.live_dashboard import LiveDashboard
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import gymnasium as gym
from collections import OrderedDict
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import json
       
class MetaPolicy(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        
        # Extract config parameters
        custom_config = model_config.get("custom_model_config", {})
        self.initial_weights = custom_config.get("initial_weights", [])
        self.num_bs = custom_config.get("num_bs", 5)
        self.num_ue = custom_config.get("num_ue", 20)
        
        # Calculate input size - one UE's observation size
        if isinstance(obs_space, gym.spaces.Dict):
            # For Dict space, we need the size of a single agent's observation
            # This should be 2*num_bs+1
            input_size = 3 * self.num_bs + 1
        else:
            input_size = np.prod(obs_space.shape)
            
        # print(f"Calculated input size: {input_size}")
        
        # Enhanced network architecture for better performance
        hidden_size = 64  # Add a hidden layer for more expressive policy
        
        # Policy network for actions - each UE chooses from num_bs actions
        self.policy_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_bs)  # Output should match number of BS options
        )
        
        # Value network (critic) - also with a hidden layer
        self.value_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Store current value function output
        self._cur_value = None
        
        # Initialize weights using metaheuristic solution if available
        if self.initial_weights:
            self._apply_initial_weights()
            
        # Print network shapes for debugging
        # print(f"Policy network: {self.policy_network}")
        # print(f"Value network: {self.value_network}")
        
    def _apply_initial_weights(self):
        """Apply initial weights to bias the policy"""
        # Each UE should have its own policy preferences
        with torch.no_grad():
            # Get the last layer of the policy network
            policy_output_layer = self.policy_network[-1]
            
            # Initialize with small random weights for exploration
            policy_output_layer.weight.data.normal_(0.0, 0.01)
            policy_output_layer.bias.data.fill_(0.0)
            
            # If we have initial weights from metaheuristic
            if isinstance(self.initial_weights, list) and len(self.initial_weights) > 0:
                # Determine which UE this policy is for based on available context
                # In MARL with parameter sharing, we can't know for sure,
                # so we use a more general approach
                
                # Count the frequency of each BS in the solution
                bs_counts = np.zeros(self.num_bs)
                for bs_idx in self.initial_weights:
                    if 0 <= bs_idx < self.num_bs:
                        bs_counts[bs_idx] += 1
                
                # Bias toward less congested BSs
                total_ues = sum(bs_counts)
                if total_ues > 0:
                    for bs_idx in range(self.num_bs):
                        # Lower allocation ratio = higher bias
                        congestion_factor = 1.0 - (bs_counts[bs_idx] / total_ues)
                        policy_output_layer.bias.data[bs_idx] = congestion_factor * 1.0 # Stronger bias toward less congested BSs
                
                # print(f"Applied metaheuristic bias based on BS congestion")
                
    def forward(self, input_dict, state, seq_lens):
        # Get observation from input dict
        obs = input_dict["obs"]
        
        # Debug: Check observation shape and values
        # print(f"Forward input shape: {obs.shape if hasattr(obs, 'shape') else 'dict'}")
        # if isinstance(obs, torch.Tensor) and obs.numel() > 0:
            # print(f"Forward input stats: min={obs.min().item():.4f}, max={obs.max().item():.4f}, "
            #     f"mean={obs.mean().item():.4f}, has_nan={torch.isnan(obs).any().item()}")
        
        # Handle different input types
        if isinstance(obs, dict) or isinstance(obs, OrderedDict):
            # Debug: Print dict keys and their shapes
            # print(f"Forward dict keys: {list(obs.keys())}")
            for k, v in obs.items():
                if hasattr(v, 'shape'):
                    print(f"  {k} shape: {v.shape}")
                    
            # In MARL, each agent should only receive its own observation
            # Convert all values to tensors and flatten if needed
            x = torch.cat([torch.tensor(v).flatten() for v in obs.values()])
        else:
            # Already a tensor
            x = obs.float() if isinstance(obs, torch.Tensor) else torch.FloatTensor(obs)
            
        # Ensure batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Debug: Check processed input tensor
        # print(f"Processed input shape: {x.shape}")
                
        # Forward passes
        logits = self.policy_network(x)
        self._cur_value = self.value_network(x).squeeze(-1)
        
        # # Debug: Check outputs
        # print(f"Logits shape: {logits.shape}, values shape: {self._cur_value.shape}")
        # print(f"Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, "
        #     f"mean={logits.mean().item():.4f}, has_nan={torch.isnan(logits).any().item()}")
        
        return logits, state

    def value_function(self):
        """Return value function output for current observation"""
        # This is required for PPO training
        assert self._cur_value is not None, "value function not calculated"
        
        # Debug: Check value function output
        # if isinstance(self._cur_value, torch.Tensor) and self._cur_value.numel() > 0:
        #     print(f"Value function stats: min={self._cur_value.min().item():.4f}, "
        #         f"max={self._cur_value.max().item():.4f}, mean={self._cur_value.mean().item():.4f}, "
        #         f"has_nan={torch.isnan(self._cur_value).any().item()}")
        
        return self._cur_value        
    
# After MetaPolicy definition
ModelCatalog.register_custom_model("meta_policy", MetaPolicy)
class VizCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        """Called at end of each training iteration."""
        it = result["training_iteration"]
        if it % 5 != 0:
            return

        env = algorithm.workers.local_worker().env
        policy = algorithm.get_policy()
        obs, _ = env.reset()
        associations = []

        for _ in range(env.episode_length):
            acts = {}
            for i in range(env.num_ue):
                o = obs[f"ue_{i}"]
                logits, _ = policy.model({"obs": o})
                acts[f"ue_{i}"] = int(logits.argmax(dim=-1).item())

            obs, _, done, _, _ = env.step(acts)
            associations.append([ue.associated_bs for ue in env.ues])
            if done["__all__"]:
                break

        # Dump JSON for Streamlit to pick up
        os.makedirs("/tmp/assoc", exist_ok=True)
        with open(f"/tmp/assoc/assoc_iter{it}.json", "w") as f:
            json.dump({"iteration": it, "associations": associations}, f)
        print(f"[VizCallback] wrote /tmp/assoc/assoc_iter{it}.json")
# RAY_DEDUP_LOGS=0
class HybridTraining:
    def __init__(self, config: Dict):
        # Initialize Ray AFTER path modification 
        # Initialize Ray with robust error handling
        try:
            if not ray.is_initialized():
                ray.init(
                    runtime_env={
                        "env_vars": {"PYTHONPATH": project_root},
                        "working_dir": project_root
                    },
                    logging_level=logging.INFO,
                    log_to_driver=True,
                    ignore_reinit_error=True,
                    **config.get("ray_resources", {})
                )
            
            # Try to verify packages are accessible but don't fail if they aren't
            try:
                @ray.remote
                def verify_package():
                    try:
                        # Print working directory and Python path for debugging
                        import os, sys
                        print(f"Current working directory: {os.getcwd()}")
                        print(f"Python path: {sys.path}")
                        
                        # Import required modules
                        from core.envs.custom_channel_env import NetworkEnvironment
                        print("✅ Successfully imported NetworkEnvironment")
                        return True
                    except ImportError as e:
                        print(f"❌ Import failed: {e}")
                        return False
                
                package_check = ray.get(verify_package.remote())
                if not package_check:
                    print("WARNING: Package verification failed, but continuing anyway...")
                
            except Exception as e:
                print(f"WARNING: Package verification error, but continuing: {e}")
        
        except Exception as e:
            print(f"Ray initialization error: {e}")
            ray.init(
                ignore_reinit_error=True,
                num_cpus=2  # Minimal fallback configuration
            )
        
        # Import NetworkEnvironment - handle both direct import and delayed import
        try:
            from core.envs.custom_channel_env import NetworkEnvironment
            self.env = NetworkEnvironment(config["env_config"])
        except ImportError:
            # Dynamic import as fallback
            import importlib
            try:
                module = importlib.import_module("core.envs.custom_channel_env")
                NetworkEnvironment = getattr(module, "NetworkEnvironment")
                self.env = NetworkEnvironment(config["env_config"])
            except Exception as e:
                raise ImportError(f"Failed to import NetworkEnvironment: {e}")    
        # @ray.remote
        # def verify_package():
        #     try:
        #         from core.envs.custom_channel_env import NetworkEnvironment
        #         from core.hybrid_trainer.hybrid_training import HybridTraining
        #         return True
        #     except ImportError as e:
        #         print(f"Import failed: {e}")
        #         return False
        # assert ray.get(verify_package.remote()), "Package verification failed!"
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
    
    def _execute_marl_phase_direct(self, initial_policy: np.ndarray = None):
        """
        A direct‐API PPO loop that after each train() does one env.step()
        so that your step() stores last_info, and then yields the solution.
        """
        print("Running the marl phase using direct-API PPO loop....")
        initial_weights = []
        if initial_policy is not None:
            initial_weights = initial_policy.tolist()  # ✅ Correct extraction
            assert len(initial_weights) == self.config["env_config"]["num_ue"]
            
        env_config = {
            **self.config["env_config"]            
        }
        # 1) build the algorithm once
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
                lr=0.0005,  # Slightly higher learning rate
                lr_schedule=[(0, 0.00005), (1000, 0.0001), (10000, 0.0005)],  # Gradual lr increase
                entropy_coeff=0.01,  # Add exploration
                kl_coeff=0.2,
                train_batch_size=4000,
                sgd_minibatch_size=128,
                num_sgd_iter=10,
                clip_param=0.2,
            ).env_runners(
            sample_timeout_s=3600, # 600,  # Increase from default (180s) to 10 minutes
            rollout_fragment_length=25 # 50  # Decrease from default (200) to collect samples faster
                )
            .multi_agent(
                policies={
                    f"ue_{i}": (None, self.obs_space[f"ue_{i}"], self.act_space[f"ue_{i}"], {})
                    for i in range(self.config["env_config"]["num_ue"])
                },
                policy_mapping_fn=lambda agent_id, episode=None, worker=None, **kwargs: agent_id
            )
        )
        algo = marl_config.build()
        # 2) (optionally) restore from a checkpoint
        #    if you wanted to warm-start from a Tune checkpoint:
        # if self.config.get("checkpoint_dir"):
        #     algo.restore(self.config["checkpoint_dir"])

        # 3) run exactly marl_steps_per_phase training iterations
        results = []
        for it in range(self.config["marl_steps_per_phase"]):
            result = algo.train()
            results.append(result)
            # call your callback so that Streamlit sees the metrics
            for cb in self._callbacks:
                cb.on_result(None, None, None, result)

        # 4) return the trained algo so that your wrapper can extract solution
        return algo, results
            
    def _execute_marl_phase(self, initial_policy: Dict = None):
        print(f"\n Starting {self.config.get('marl_algorithm','PPO').upper()} training...")

        # Prepare the list of initial biases for the policy network
        initial_weights = []
        if initial_policy is not None:
            initial_weights = initial_policy.tolist()
            assert len(initial_weights) == self.config["env_config"]["num_ue"]

        env_config = {**self.config["env_config"]}

        # Build the PPO config
        marl_config = (
            PPOConfig()
            .environment("NetworkEnv", env_config=env_config)
            .env_runners(
                rollout_fragment_length=10,   # collect 10 frames per worker
                sample_timeout_s=600          # 10 minutes max
            )
            .training(
                model={
                    "custom_model": "meta_policy",
                    "custom_model_config": {
                        "initial_weights": initial_weights,
                        "num_bs": self.config["env_config"]["num_bs"],
                        "num_ue": self.config["env_config"]["num_ue"],
                    }
                },
                gamma=0.99,
                lr=5e-4,
                lr_schedule=[(0, 5e-5), (1000, 1e-4), (10000, 5e-4)],
                entropy_coeff=0.01,
                kl_coeff=0.2,
                train_batch_size=4000,
                sgd_minibatch_size=128,
                num_sgd_iter=10,
                clip_param=0.2,
            )
            .multi_agent(
                policies={
                    f"ue_{i}": (None,
                                self.obs_space[f"ue_{i}"],
                                self.act_space[f"ue_{i}"], {})
                    for i in range(self.config["env_config"]["num_ue"])
                },
                policy_mapping_fn=lambda agent_id, episode=None, worker=None, **kwargs: agent_id
            )
            # Register our VizCallback so RLlib will call it each iteration
            .callbacks(VizCallback)
        )

        analysis = ray.tune.run(
            "PPO",
            config=marl_config.to_dict(),
            stop={"training_iteration": self.config["marl_steps_per_phase"]},
            checkpoint_at_end=True,
        )
        print("Trial errors:", analysis.errors)
        return analysis
        # Add this after tune.run()
        

    def _create_marl_callback(self):
        """Create a Ray Tune callback for tracking training progress"""
        
        class MALRCallback(ray.tune.Callback):
            def __init__(self, parent):
                self.parent = parent
                
            def on_trial_result(self, iteration, trials, trial, result, **info):
                # Get the actual reward values from result dictionary
                reward_mean = result.get("env_runners/episode_return_mean", 0)
                reward_min = result.get("env_runners/episode_return_min", 0)
                reward_max = result.get("env_runners/episode_return_max", 0)
                episode_len = result.get("env_runners/episode_len_mean", 0)

                print(f"[Trial {trial.trial_id}] Iter {result['training_iteration']} | "
                    f"Reward: {reward_mean:.3f} | "  # Use the correctly retrieved value
                    f"Length: {episode_len:.1f}")
                
                if self.parent.kpi_logger and self.parent.kpi_logger.enabled:
                    metrics = {
                        "iteration": self.parent.current_epoch * self.parent.config["marl_steps_per_phase"] + 
                                result["training_iteration"],
                        "reward_mean": reward_mean,  # Use the correctly retrieved value
                        "reward_min": reward_min, 
                        "reward_max": reward_max,
                        "episode_length": episode_len
                    }
                    
                    self.parent.kpi_logger.log_metrics(
                        phase="marl",
                        algorithm="PPO",
                        metrics=metrics,
                        episode=result.get("training_iteration", 0)
                    )
                
        return MALRCallback(self)
    

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
        "metaheuristic": "pfo",
        "comparison_mode": False,
        "metaheuristic_algorithms": ["pfo", "co", "coa", "do", "fla", "gto", "hba", "hoa", "avoa","aqua", "poa", "rime", "roa", "rsa", "sto"], #
        "marl_algorithm": "PPO", 
        
        # Environment parameters
        "env_config": {
            "num_bs": 10,
            "num_ue": 200,
            "episode_length": 10,
            "log_kpis": True
        },
        
                        
        # Training parameters
        "max_epochs": 50,
        "marl_steps_per_phase": 1,
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
