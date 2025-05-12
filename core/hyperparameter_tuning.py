# hyperparameter_tuning.py
import sys
import os

# Configure project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

from typing import Dict, Type, Callable
from tqdm import tqdm
import time
import pandas as pd
from core.envs.custom_channel_env import NetworkEnvironment
from functools import partial
from core.algorithms.coa import COAOptimization
from core.algorithms.co import CheetahOptimization
from core.algorithms.do import DandelionOptimization
from core.algorithms.gto import GTOOptimization
from core.algorithms.hba import HBAOptimization
from core.algorithms.rsa import RSAOptimization
from core.algorithms.sto import STOOptimization
from core.algorithms.poa import PelicanOptimization
from core.algorithms.hoa import HippoOptimization
from core.algorithms.fla import FLAOptimization
from core.algorithms.rime import RIMEOptimization
from core.algorithms.avoa import AVOAOptimization
from core.algorithms.aqua import AquilaOptimization
from core.algorithms.pfo import PolarFoxOptimization
from core.algorithms.roa import RainbowOptimization

import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_param_importances, plot_parallel_coordinate
class OptunaTuner:
    def __init__(self, env: NetworkEnvironment):
        self.env = env
        self.studies = {}      
    
    
    @staticmethod
    def suggest_avoaparams(trial):
        return {
            'vultures': trial.suggest_int('vultures', 20, 50),
            'exploration_rate': trial.suggest_float('exploration_rate', 0.3, 0.9),
            'exploitation_rate': trial.suggest_float('exploitation_rate', 0.5, 1.0),
            'satiety_threshold': trial.suggest_float('satiety_threshold', 0.1, 0.5),
            'alpha': trial.suggest_float('alpha', 1.0, 3.0),
            'beta': trial.suggest_float('beta', 1.0, 2.5)
        }

    @staticmethod
    def suggest_cheetahoptimization_params(trial):
        return {
            'cheetahs': trial.suggest_int('cheetahs', 20, 50),
            'sprint_prob': trial.suggest_float('sprint_prob', 0.4, 0.8),
            'rest_threshold': trial.suggest_float('rest_threshold', 0.2, 0.4),
            'acceleration': trial.suggest_float('acceleration', 1.0, 1.5),
            'energy_decay': trial.suggest_float('energy_decay', 0.9, 0.99)
        }

    @staticmethod
    def suggest_coaoptimization_params(trial):
        return {
            'coatis': trial.suggest_int('coatis', 20, 50),
            'attack_rate': trial.suggest_float('attack_rate', 0.05, 0.3),
            'explore_rate': trial.suggest_float('explore_rate', 0.5, 0.95)
        }

    @staticmethod
    def suggest_dandelionoptimization_params(trial):
        return {
            'seeds': trial.suggest_int('seeds', 20, 50),
            'wind_factor': trial.suggest_float('wind_factor', 0.3, 0.8),
            'lift_coeff': trial.suggest_float('lift_coeff', 1.0, 1.5),
            'descent_rate': trial.suggest_float('descent_rate', 0.85, 0.99)
        }

    @staticmethod
    def suggest_flaoptimization_params(trial):
        return {
            'particles': trial.suggest_int('particles', 20, 50),
            'diffusion_rate': trial.suggest_float('diffusion_rate', 0.5, 0.95),
            'random_walk_prob': trial.suggest_float('random_walk_prob', 0.2, 0.6),
            'time_step': trial.suggest_float('time_step', 0.3, 0.7),
            'decay_factor': trial.suggest_float('decay_factor', 0.9, 0.99)
        }

    @staticmethod
    def suggest_gtooptimization_params(trial):
        return {
            'gorillas': trial.suggest_int('gorillas', 20, 50),
            'silverback_influence': trial.suggest_float('silverback_influence', 0.6, 0.95),
            'migration_prob': trial.suggest_float('migration_prob', 0.2, 0.5),
            'social_factor': trial.suggest_float('social_factor', 0.3, 0.8)
        }

    @staticmethod
    def suggest_hbaoptimization_params(trial):
        return {
            'badgers': trial.suggest_int('badgers', 20, 50),
            'intensity': trial.suggest_float('intensity', 0.7, 1.0),
            'density_factor': trial.suggest_float('density_factor', 0.8, 1.2),
            'honey_prob': trial.suggest_float('honey_prob', 0.3, 0.7)
        }

    @staticmethod
    def suggest_hippooptimization_params(trial):
        return {
            'pod_size': trial.suggest_int('pod_size', 20, 50),
            'aggression_rate': trial.suggest_float('aggression_rate', 0.2, 0.5),
            'social_factor': trial.suggest_float('social_factor', 0.4, 0.8),
            'yawn_impact': trial.suggest_float('yawn_impact', 0.3, 0.6),
            'territorial_decay': trial.suggest_float('territorial_decay', 0.85, 0.99)
        }

    @staticmethod
    def suggest_pelicanoptimization_params(trial):
        return {
            'pelicans': trial.suggest_int('pelicans', 20, 50),
            'initial_movement': trial.suggest_float('initial_movement', 0.6, 0.9),
            'scoop_intensity': trial.suggest_float('scoop_intensity', 0.1, 0.4),
            'decay_factor': trial.suggest_float('decay_factor', 0.9, 0.99)
        }

    @staticmethod
    def suggest_polarfoxoptimization_params(trial):
        return {
            'population_size': trial.suggest_int('population_size', 30, 60),
            'mutation_factor': trial.suggest_float('mutation_factor', 0.1, 0.5),
            'jump_rate': trial.suggest_float('jump_rate', 0.1, 0.4),
            'follow_rate': trial.suggest_float('follow_rate', 0.2, 0.5),
            'group_weights': [
                trial.suggest_float(f'group_weight_{i}', 0.1, 1.0) 
                for i in range(4)
            ]
        }

    @staticmethod
    def suggest_rimeoptimization_params(trial):
        return {
            'particles': trial.suggest_int('particles', 20, 50),
            'temperature': trial.suggest_float('temperature', 0.8, 1.2),
            'cooling_rate': trial.suggest_float('cooling_rate', 0.9, 0.99),
            'stability_threshold': trial.suggest_float('stability_threshold', 0.05, 0.2),
            'phase_ratio': trial.suggest_float('phase_ratio', 0.4, 0.8)
        }

    @staticmethod
    def suggest_rainbowoptimization_params(trial):
        return {
            'rays': trial.suggest_int('rays', 20, 50),
            'refraction_rate': trial.suggest_float('refraction_rate', 0.5, 0.9),
            'dispersion_factor': trial.suggest_float('dispersion_factor', 0.3, 0.7),
            'prism_effect': trial.suggest_float('prism_effect', 1.0, 1.5)
        }

    @staticmethod
    def suggest_rsaoptimization_params(trial):
        return {
            'reptiles': trial.suggest_int('reptiles', 20, 50),
            'alpha': trial.suggest_float('alpha', 0.05, 0.3),
            'beta': trial.suggest_float('beta', 1.2, 2.0),
            'hunting_prob': trial.suggest_float('hunting_prob', 0.6, 0.9)
        }

    @staticmethod
    def suggest_stooptimization_params(trial):
        return {
            'tigers': trial.suggest_int('tigers', 20, 50),
            'territory_radius': trial.suggest_float('territory_radius', 0.3, 0.6),
            'attack_intensity': trial.suggest_float('attack_intensity', 1.2, 2.0),
            'marking_rate': trial.suggest_float('marking_rate', 0.1, 0.4)
        }
    # Add similar methods for other algorithms...

class BatchTuner:
    def __init__(self, env, n_trials=100, timeout=3600):
        self.env = env
        self.n_trials = n_trials
        self.timeout = timeout
        self.results = []
        self.tuner = OptunaTuner(env)
        
    def run_all(self):
        algorithms = [
            ("AVOA", AVOAOptimization),
            ("Cheetah", CheetahOptimization),
            ("COA", COAOptimization),
            ("Dandelion", DandelionOptimization),
            ("FLA", FLAOptimization),
            ("GTO", GTOOptimization),
            ("HBA", HBAOptimization),
            ("Hippo", HippoOptimization),
            ("Pelican", PelicanOptimization),
            ("PolarFox", PolarFoxOptimization),
            ("RIME", RIMEOptimization),
            ("Rainbow", RainbowOptimization),
            ("RSA", RSAOptimization),
            ("STO", STOOptimization)
        ]

        for algo_name, algo_class in algorithms:
            print(f"\n{'='*40}")
            print(f"Starting optimization for {algo_name}")
            print(f"{'='*40}")
            
            try:
                study = self.run_study(algo_class)
                self.store_results(algo_name, study)
                self.save_visualizations(algo_name, study)
                self.print_summary(algo_name, study)
            except Exception as e:
                print(f"Error optimizing {algo_name}: {str(e)}")
                continue

        self.final_report()

    def run_study(self, algorithm_class):
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler()
        )
        
        def objective(trial):
            common_params = {
                'iterations': trial.suggest_int('iterations', 10, 50),
                'population_size': trial.suggest_int('population_size', 20, 100)
            }
            
            algo_method = getattr(
                self.tuner.__class__, 
                f'suggest_{algorithm_class.__name__.lower()}_params',
                lambda _: {}
            )
            algo_params = algo_method(trial)
            
            optimizer = algorithm_class(self.env, **common_params, **algo_params)
            result = optimizer.run()
            
            for i, fitness in enumerate(result['agents']['fitness']):
                trial.report(fitness, i)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                    
            return result['metrics']['fitness']

        study.optimize(
            objective, 
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        return study
    def store_results(self, algo_name, study):
            self.results.append({
                'algorithm': algo_name,
                'best_value': study.best_value,
                'best_params': study.best_params,
                'duration': study._stopwatch.duration
            })

    def save_visualizations(self, algo_name, study):
            # Save visualizations to files
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html(f"{algo_name}_param_importance.html")
            
            fig2 = optuna.visualization.plot_parallel_coordinate(study)
            fig2.write_html(f"{algo_name}_parallel_coord.html")

    def print_summary(self, algo_name, study):
            print(f"\n{algo_name} Optimization Complete!")
            print(f"Best Fitness: {study.best_value:.4f}")
            print(f"Best Parameters:")
            for k, v in study.best_params.items():
                print(f"  {k}: {v}")
            print(f"Duration: {study._stopwatch.duration:.2f} seconds\n")

    def final_report(self):
            print("\n\nFINAL OPTIMIZATION REPORT")
            print("="*50)
            df = pd.DataFrame(self.results)
            print(df[['algorithm', 'best_value', 'duration']])
            df.to_csv("optimization_results.csv", index=False)
            print("\nDetailed results saved to optimization_results.csv")

if __name__ == "__main__":
    # Initialize network environment
    env = NetworkEnvironment(num_bs=10, num_ue=50)
    
    # Configure tuner (100 trials per algorithm, 1 hour max per algorithm)
    tuner = BatchTuner(env, n_trials=100, timeout=3600)
    
    # Start batch optimization
    print("Starting batch optimization of all algorithms...")
    start_time = time.time()
    tuner.run_all()
    duration = time.time() - start_time
    
    print(f"\nTotal optimization time: {duration/3600:.2f} hours")