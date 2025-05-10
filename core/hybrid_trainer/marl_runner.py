from typing import List, Optional, Dict, Generator, Tuple, Any
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import os
import sys

# Add the project root to Python path (important fix)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to PYTHONPATH: {project_root}")

# Import your environment and other dependencies
from core.envs.custom_channel_env import NetworkEnvironment
from core.hybrid_trainer.kpi_logger import KPITracker
from core.hybrid_trainer.hybrid_training import HybridTraining

# Register environment with Ray
def _register_environment():
    def _env_creator(env_config):
        return NetworkEnvironment(env_config)
    
    try:
        register_env("NetworkEnv", _env_creator)
    except Exception:
        # Already registered
        pass

def run_marl(
    env_config: Dict,
    ray_resources: Dict,
    initial_solution: Optional[List[int]] = None,
    marl_steps_per_epoch: int = 1,
    total_epochs: int = 50,
    checkpoint_dir: Optional[str] = None,
    model_config: Optional[Dict] = None,
) -> Generator[Tuple[Dict[str, float], List[int]], None, None]:
    """
    Generator that runs PPO in HybridTraining, warm-started by a metaheuristic.
    
    Yields (metrics, solution) each epoch for live plotting in Streamlit.
    
    Args:
        env_config: Configuration for the NetworkEnvironment
        ray_resources: Resources to allocate via Ray
        initial_solution: Optional metaheuristic solution to warm-start
        marl_steps_per_epoch: Number of PPO iterations per epoch
        total_epochs: Total number of epochs to run
        checkpoint_dir: Directory to save checkpoints (optional)
        model_config: Additional model configuration (optional)
        
    Yields:
        Tuple of (metrics dict, solution list)
    """
    # Ensure no prior Ray instance is active
    try:
        ray.shutdown()
    except Exception:
        pass
    
    # (Re)initialize Ray with proper error handling and explicit path configuration
    try:
        ray.init(
            ignore_reinit_error=True,
            runtime_env={
                "env_vars": {"PYTHONPATH": project_root},
                "working_dir": project_root
            },
            **ray_resources
        )
    except Exception as e:
        print(f"Ray initialization error: {e}")
        # Fallback to minimal resources
        ray.init(
            ignore_reinit_error=True,
            runtime_env={
                "env_vars": {"PYTHONPATH": project_root},
                "working_dir": project_root
            },
            num_cpus=2
        )
    
    # Register environment
    _register_environment()
    
    # Build hybrid training config
    hybrid_config = {
        "env_config": env_config,
        "marl_algorithm": "PPO",
        "marl_steps_per_phase": marl_steps_per_epoch,
        "ray_resources": ray_resources,
        "logging": {"enabled": False},
    }
    
    if checkpoint_dir:
        hybrid_config["checkpoint_dir"] = checkpoint_dir
    
    # Create HybridTraining instance
    try:
        # Verify package imports directly before passing to HybridTraining
        @ray.remote
        def verify_imports():
            try:
                import os, sys
                print(f"Current working directory: {os.getcwd()}")
                print(f"Python path: {sys.path}")
                from core.envs.custom_channel_env import NetworkEnvironment
                from core.hybrid_trainer.hybrid_training import HybridTraining
                print("✅ Successfully imported required modules!")
                return True
            except ImportError as e:
                print(f"❌ Import failed: {e}")
                return False
        
        if not ray.get(verify_imports.remote()):
            print("WARNING: Package verification failed, but continuing anyway...")
        
        hybrid = HybridTraining(hybrid_config)
    except Exception as e:
        print(f"HybridTraining initialization error: {e}")
        raise
    
    # Ensure initial_solution is properly formatted if provided
    policy = None
    if initial_solution is not None:
        if isinstance(initial_solution, list) and len(initial_solution) == env_config["num_ue"]:
            policy = np.array(initial_solution, dtype=np.int32)
            print(f"Using initial solution with {len(policy)} UE assignments")
        else:
            print(f"Warning: Invalid initial_solution format. Expected list of length {env_config['num_ue']}")
    
    # Main training loop
    for epoch in range(total_epochs):
        print(f"\nEpoch {epoch+1}/{total_epochs}")
        
        try:
            # Run exactly marl_steps_per_epoch PPO iterations
            print("→ about to run marl phase", epoch)
            analysis = hybrid._execute_marl_phase(initial_policy=policy)
            # analysis = hybrid._execute_marl_phase_direct(initial_policy=policy)
            print("← returned from marl phase", epoch)
            # Extract metrics with error handling
            if len(analysis.trials) == 0:
                print("Warning: No trials completed in analysis")
                metrics = {
                    "reward_mean": 0,
                    "reward_min": 0,
                    "reward_max": 0,
                    "epoch": epoch
                }
            else:
                trial = analysis.trials[0]
                res = trial.last_result
                metrics = {
                    "reward_mean": res.get("env_runners/episode_return_mean", 0),
                    "reward_min": res.get("env_runners/episode_return_min", 0),
                    "reward_max": res.get("env_runners/episode_return_max", 0),
                    "episode_length": res.get("env_runners/episode_len_mean", 0),
                    "epoch": epoch
                }
            
            last_info = hybrid.env.get_last_info()
            if last_info and "__common__" in last_info and "current solution" in last_info["__common__"]:
                solution = last_info["__common__"]["current solution"]
            else:
                solution = _extract_solution(hybrid, analysis)
            # —— end new block ——

            # sanitize solution, yield, warm-start next epoch …
            solution = [bs if 0 <= bs < env_config["num_bs"] else 0 for bs in solution]
            yield metrics, solution
            # policy = np.array(solution, dtype=np.int32)
            
            # Use latest solution as next warm-start
            if solution:
                policy = np.array(solution, dtype=np.int32)
            
        except Exception as e:
            print(f"Error in epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            # Return empty metrics and previous solution to continue
            yield {"reward_mean": 0, "reward_min": 0, "reward_max": 0, "error": str(e)}, \
                  solution if 'solution' in locals() else []
    
    
    # Clean up Ray resources
    ray.shutdown()

def _extract_solution(hybrid, analysis):
    """
    Extract UE -> BS association solution from the trained policy.
    
    This function performs a greedy rollout and retrieves the association decisions.
    
    Args:
        hybrid: HybridTraining instance
        analysis: Ray Tune analysis object
        
    Returns:
        List of BS indices for each UE
    """
    try:
        # First, try to get from environment's last info
        env = hybrid.env
        if hasattr(env, 'get_last_info'):
            env_info = env.get_last_info()
            print("<<<< env_info from get_last_info():", env_info)
            if env_info and "__common__" in env_info and "current_solution" in env_info["__common__"]:
                print("Found solution in env info!")
                return env_info["__common__"]["current_solution"]
        
        # Otherwise, try to extract from current UE state
        if hasattr(env, 'ues'):
            print("Extracting solution from UE state")
            solution = []
            for ue in env.ues:
                if hasattr(ue, 'associated_bs') and ue.associated_bs is not None:
                    solution.append(ue.associated_bs)
                else:
                    # Default to BS 0 for unassociated UEs
                    solution.append(0)
            return solution
            
        # Last resort: create an empty solution
        print("Returning default solution")
        return [0] * env.num_ue
        
    except Exception as e:
        print(f"Error extracting solution: {e}")
        # Return empty list with appropriate length if we can determine it
        if hasattr(hybrid.env, 'num_ue'):
            return [0] * hybrid.env.num_ue
        return []

