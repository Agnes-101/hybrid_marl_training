# hyperparameter_tuning.py
import sys
import os

# Configure project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

import optuna
from typing import Dict, Type, Callable
from functools import partial
from envs.custom_channel_env import NetworkEnvironment
from core.algorithms.old.aco import ACO
from core.algorithms.old.bat import BatOptimization
from core.algorithms.old.cs import CSOptimization
from core.algorithms.old.de import DEOptimization
from core.algorithms.old.fa import FireflyOptimization
from core.algorithms.old.ga import GAOptimization
from core.algorithms.old.gwo import GWOOptimization
from core.algorithms.old.hs import HarmonySearchOptimization
from core.algorithms.old.ica import ICAOptimization
from algorithms.pfo import PolarFoxOptimization
from core.algorithms.old.pso import PSOOptimization
from core.algorithms.old.sa import SAOptimization
from core.algorithms.old.tabu import TabuSearchOptimization
from core.algorithms.old.woa import WOAOptimization

class MetaheuristicTuner:
    def __init__(self, env: NetworkEnvironment, num_trials=100):
        self.env = env
        self.num_trials = num_trials
        self.studies: Dict[str, optuna.Study] = {}

    def _objective(self, trial: optuna.Trial, 
                 algorithm_class: Type,
                 get_params: Callable) -> float:
        """Universal objective function for all algorithms"""
        params = get_params(trial)
        algorithm = algorithm_class(env=self.env, **params)
        result = algorithm.run(visualize_callback=None)  # Disable visualization during tuning
        return result["metrics"]["fitness"]

    def tune_algorithm(self, algorithm_name: str,
                      algorithm_class: Type,
                      get_params: Callable) -> Dict:
        """Tune a specific algorithm"""
        study = optuna.create_study(direction="maximize")
        objective = partial(self._objective, 
                          algorithm_class=algorithm_class,
                          get_params=get_params)
        study.optimize(objective, n_trials=self.num_trials)
        self.studies[algorithm_name] = study
        return study.best_params

    # Parameter space definitions for each algorithm
    @staticmethod
    def get_de_params(trial: optuna.Trial) -> Dict:
        return {
            "population_size": trial.suggest_int("de_pop_size", 20, 100),
            "F": trial.suggest_float("de_F", 0.1, 2.0),
            "CR": trial.suggest_float("de_CR", 0.1, 0.99)
        }

    @staticmethod
    def get_pso_params(trial: optuna.Trial) -> Dict:
        return {
            "swarm_size": trial.suggest_int("pso_swarm_size", 20, 100),
            "c1": trial.suggest_float("pso_c1", 0.1, 3.0),
            "c2": trial.suggest_float("pso_c2", 0.1, 3.0),
            "w": trial.suggest_float("pso_w", 0.1, 1.0)
        }

    @staticmethod
    def get_ga_params(trial: optuna.Trial) -> Dict:
        return {
            "population_size": trial.suggest_int("ga_pop_size", 20, 100),
            "mutation_rate": trial.suggest_float("ga_mutation", 0.01, 0.5)
        }

    # Add similar parameter definitions for other algorithms...

    def tune_all(self) -> Dict[str, Dict]:
        """Tune all implemented algorithms"""
        return {
            "DE": self.tune_algorithm("DE", DEOptimization, self.get_de_params),
            "PSO": self.tune_algorithm("PSO", PSOOptimization, self.get_pso_params),
            "GA": self.tune_algorithm("GA", GAOptimization, self.get_ga_params),
            # Add other algorithms...
        }

    def get_best_configs(self) -> Dict[str, Dict]:
        """Retrieve best configurations after tuning"""
        return {name: study.best_params for name, study in self.studies.items()}

if __name__ == "__main__":
    # Usage example
    from envs.custom_channel_env import NetworkEnvironment
    
    # Initialize your network environment
    env = NetworkEnvironment(...)
    
    tuner = MetaheuristicTuner(env=env, num_trials=50)
    
    # Tune all algorithms
    best_configs = tuner.tune_all()
    
    # Or tune specific algorithm
    de_best = tuner.tune_algorithm(
        "DE", 
        DEOptimization,
        MetaheuristicTuner.get_de_params
    )
    
    print("Best configurations:", best_configs)