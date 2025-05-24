# import numpy as np
# from envs.custom_channel_env import NetworkEnvironment

# class HBAOptimization:
#     def __init__(self, env: NetworkEnvironment,iterations=20,badgers = 30,intensity = 0.9,density_factor = 1.0,
#                 honey_prob = 0.5,kpi_logger=None):
#         """Honey Badger Algorithm for 6G user association optimization"""
#         self.env = env
#         self.badgers = badgers      # Population size
#         self.iterations = iterations
#         self.intensity = intensity  # Smell intensity coefficient
#         self.density_factor = density_factor  # Exploration density
#         self.honey_prob = honey_prob  # Probability of honey phase
#         self.seed = 42
#         self.kpi_logger = kpi_logger
        
#         # Visualization and tracking
#         self.positions = np.empty((0, 2))  # (mean_assoc, std_assoc)
#         self.fitness_history = []
#         self.rng = np.random.RandomState(self.seed)
#         self.best_solution = None

#     def run(self, visualize_callback=None, kpi_logger=None) -> dict:
#         """Main optimization process"""
#         original_state = self.env.get_state_snapshot()
#         num_ue = self.env.num_ue
#         num_bs = self.env.num_bs
        
#         # Initialize badger population
#         solutions = np.array([self._random_solution(num_ue, num_bs) 
#                             for _ in range(self.badgers)])
#         best_fitness = -np.inf
#         self.best_solution = solutions[0].copy()

#         for iteration in range(self.iterations):
#             # Adaptive parameter adjustment
#             decay_factor = 1 - (iteration / self.iterations)
#             current_density = self.density_factor * decay_factor
            
#             # Update badger positions
#             solutions = self._update_positions(solutions, current_density)
            
#             # Evaluate fitness
#             fitness_values = np.array([self.env.evaluate_detailed_solution(sol)["fitness"] 
#                                     for sol in solutions])
#             current_best_idx = np.argmax(fitness_values)
            
#             # Update best solution
#             if fitness_values[current_best_idx] > best_fitness:
#                 best_fitness = fitness_values[current_best_idx]
#                 self.best_solution = solutions[current_best_idx].copy()

#             # Logging and visualization
#             # self._update_visualization(iteration)
#             current_metrics = self.env.evaluate_detailed_solution(self.best_solution)
            
#             if self.kpi_logger:
#                 self.kpi_logger.log_metrics(
#                     episode=iteration,
#                     phase="metaheuristic",
#                     algorithm="hba",
#                     metrics=current_metrics
#                 )
#                 print(f"HBA Iter {iteration}: Best Fitness = {best_fitness:.4f}")

#             if visualize_callback:
#                 viz_metrics = {
#                     "fitness": current_metrics["fitness"],
#                     "average_sinr": current_metrics["average_sinr"],
#                     "fairness": current_metrics["fairness"]
#                 }
#                 visualize_callback(viz_metrics, self.best_solution)

#         # Finalize and return results
#         self.env.set_state_snapshot(original_state)
#         self.env.apply_solution(self.best_solution)
        
#         return {
#             "solution": self.best_solution,
#             "metrics": self.env.evaluate_detailed_solution(self.best_solution),
#             "agents": {
#                 "positions": self.positions.tolist(),
#                 "fitness": self.fitness_history,
#                 "algorithm": "hba"
#             }
#         }

#     def _random_solution(self, num_ue: int, num_bs: int) -> np.ndarray:
#         """Generate random UE-BS associations"""
#         return self.rng.randint(0, num_bs, size=num_ue)

#     def _update_positions(self, solutions: np.ndarray, density: float) -> np.ndarray:
#         """Update positions using digging and honey phases"""
#         new_solutions = []
#         num_bs = self.env.num_bs
#         fitness_values = np.array([self.env.evaluate_detailed_solution(sol)["fitness"] 
#                                  for sol in solutions])
        
#         # Normalize fitness values for intensity calculation
#         max_fitness = np.max(fitness_values)
#         min_fitness = np.min(fitness_values)
#         normalized_fitness = (fitness_values - min_fitness) / (max_fitness - min_fitness + 1e-10)
        
#         for idx, sol in enumerate(solutions):
#             if self.rng.rand() < self.honey_prob:
#                 # Honey phase: move toward best solution
#                 new_sol = self._honey_phase(sol, density)
#             else:
#                 # Digging phase: local exploration
#                 new_sol = self._digging_phase(sol, normalized_fitness[idx], density)
            
#             new_solutions.append(new_sol)
            
#         return np.array(new_solutions)

#     def _digging_phase(self, current_sol: np.ndarray, intensity: float, density: float) -> np.ndarray:
#         """Exploration phase with smell intensity guidance"""
#         new_sol = current_sol.copy()
#         num_ue = len(new_sol)
        
#         for ue in range(num_ue):
#             if self.rng.rand() < density:
#                 # Random exploration weighted by intensity
#                 delta = int(self.intensity * density * self.rng.normal(0, self.env.num_bs/2))
#                 new_bs = (current_sol[ue] + delta) % self.env.num_bs
#                 new_sol[ue] = new_bs
#         return new_sol

#     def _honey_phase(self, current_sol: np.ndarray, density: float) -> np.ndarray:
#         """Exploitation phase moving toward best solution"""
#         new_sol = current_sol.copy()
#         best_sol = self.best_solution
        
#         for ue in range(len(new_sol)):
#             if self.rng.rand() < self.intensity:
#                 # Guided movement toward best solution
#                 if current_sol[ue] != best_sol[ue]:
#                     step = 1 if best_sol[ue] > current_sol[ue] else -1
#                     new_sol[ue] = (current_sol[ue] + step) % self.env.num_bs
#                 else:
#                     # Local refinement
#                     new_sol[ue] = (current_sol[ue] + self.rng.randint(-1, 2)) % self.env.num_bs
#             elif self.rng.rand() < density:
#                 # Random component
#                 new_sol[ue] = self.rng.randint(0, self.env.num_bs)
                
#         return new_sol

#     def _update_visualization(self, iteration: int):
#         """Track solution diversity metrics"""
#         current_solutions = self.env.get_current_associations()
#         mean_assoc = np.mean(current_solutions)
#         std_assoc = np.std(current_solutions)
        
#         self.positions = np.vstack([self.positions, [mean_assoc, std_assoc]])
#         self.fitness_history.append(
#             self.env.evaluate_detailed_solution(self.best_solution)["fitness"]
#         )
import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class HBAOptimization:
    """Honey Badger Algorithm for 6G user association optimization with nearest-cell seeding"""
    def __init__(
        self,
        env: NetworkEnvironment,
        iterations: int = 20,
        badgers: int = 30,
        intensity: float = 0.9,
        density_factor: float = 1.0,
        honey_prob: float = 0.5,
        init_heuristic_prob: float = 0.2,
        kpi_logger=None
    ):
        # Environment and parameters
        self.env = env
        self.iterations = iterations
        self.badgers = badgers
        self.intensity = intensity
        self.density_factor = density_factor
        self.honey_prob = honey_prob
        self.init_heuristic_prob = init_heuristic_prob
        self.kpi_logger = kpi_logger

        # RNG for reproducibility
        self.rng = np.random.RandomState(42)

        # Initialize population with heuristic seeding
        self.solutions = np.array([
            self._initial_solution() for _ in range(self.badgers)
        ])

        # Tracking
        self.best_solution = None
        self.positions = np.empty((0, 2))
        self.fitness_history = []

    def _initial_solution(self) -> np.ndarray:
        """20% chance to use nearest-cell heuristic, else random"""
        num_ue = self.env.num_ue
        num_bs = self.env.num_bs
        if self.rng.rand() < self.init_heuristic_prob:
            # Gather UE positions
            if hasattr(self.env, 'ues'):
                user_positions = [ue.position for ue in self.env.ues]
            elif hasattr(self.env, 'users'):
                user_positions = [ue.position for ue in self.env.users]
            elif hasattr(self.env, 'user_positions'):
                user_positions = self.env.user_positions
            else:
                # Fallback to random
                return self.rng.randint(0, num_bs, size=num_ue)

            # Compute nearest cell for each UE
            return np.array([self.find_nearest_cell(pos) for pos in user_positions])
        else:
            # Random assignment
            return self.rng.randint(0, num_bs, size=num_ue)

    def find_nearest_cell(self, position) -> int:
        """Find index of the nearest base station to a UE position"""
        pos = np.array(position, dtype=np.float32)
        cell_positions = np.stack([bs.position for bs in self.env.base_stations])
        distances = np.linalg.norm(cell_positions - pos, axis=1)
        return int(np.argmin(distances))

    def run(self, visualize_callback=None) -> dict:
        """Execute the HBA optimization loop"""
        # Snapshot and restore
        original_state = self.env.get_state_snapshot()

        # Evaluate initial fitnesses
        fitness_values = np.array([
            self.env.evaluate_detailed_solution(sol)["fitness"]
            for sol in self.solutions
        ])
        best_idx = np.argmax(fitness_values)
        best_fitness = fitness_values[best_idx]
        self.best_solution = self.solutions[best_idx].copy()

        # Main loop
        for iteration in range(self.iterations):
            # Linear decay for density
            decay_factor = 1 - (iteration / self.iterations)
            current_density = self.density_factor * decay_factor

            # Generate new solutions
            new_solutions = self._update_positions(self.solutions, current_density)
            self.solutions = new_solutions

            # Batch evaluate
            fitness_values = np.array([
                self.env.evaluate_detailed_solution(sol)["fitness"]
                for sol in self.solutions
            ])

            # Update best
            idx = np.argmax(fitness_values)
            if fitness_values[idx] > best_fitness:
                best_fitness = fitness_values[idx]
                self.best_solution = self.solutions[idx].copy()

            # Logging
            self.fitness_history.append(best_fitness)
            current_metrics = self.env.evaluate_detailed_solution(self.best_solution)
            if self.kpi_logger:
                self.kpi_logger.log_metrics(
                    episode=iteration,
                    phase="metaheuristic",
                    algorithm="hba",
                    metrics=current_metrics
                )

            # Visualization
            if visualize_callback:
                viz = {
                    "fitness": current_metrics["fitness"],
                    "average_sinr": current_metrics.get("average_sinr", 0),
                    "fairness": current_metrics.get("fairness", 0)
                }
                visualize_callback(viz, self.best_solution)

        # Restore and apply best
        self.env.set_state_snapshot(original_state)
        self.env.apply_solution(self.best_solution)

        return {
            "solution": self.best_solution,
            "metrics": self.env.evaluate_detailed_solution(self.best_solution),
            "agents": {
                "positions": self.positions.tolist(),
                "fitness": self.fitness_history,
                "algorithm": "hba"
            }
        }

    def _update_positions(self, solutions: np.ndarray, density: float) -> np.ndarray:
        """Update population via digging and honey phases"""
        num = len(solutions)
        # Evaluate all fitnesses once
        fitness = np.array([
            self.env.evaluate_detailed_solution(sol)["fitness"]
            for sol in solutions
        ])
        max_f, min_f = fitness.max(), fitness.min()
        norm_f = (fitness - min_f) / (max_f - min_f + 1e-10)

        updated = []
        for idx, sol in enumerate(solutions):
            if self.rng.rand() < self.honey_prob:
                new_sol = self._honey_phase(sol, density)
            else:
                new_sol = self._digging_phase(sol, norm_f[idx], density)
            updated.append(new_sol)
        return np.array(updated)

    def _digging_phase(self, sol: np.ndarray, intensity: float, density: float) -> np.ndarray:
        new_sol = sol.copy()
        num_ue = len(sol)
        for ue in range(num_ue):
            if self.rng.rand() < density:
                delta = int(self.intensity * density * self.rng.normal(0, self.env.num_bs/2))
                new_sol[ue] = (sol[ue] + delta) % self.env.num_bs
        return new_sol

    def _honey_phase(self, sol: np.ndarray, density: float) -> np.ndarray:
        new_sol = sol.copy()
        for ue in range(len(sol)):
            if self.rng.rand() < self.intensity:
                # move toward best solution
                if sol[ue] != self.best_solution[ue]:
                    step = 1 if self.best_solution[ue] > sol[ue] else -1
                    new_sol[ue] = (sol[ue] + step) % self.env.num_bs
                else:
                    new_sol[ue] = (sol[ue] + self.rng.randint(-1, 2)) % self.env.num_bs
            elif self.rng.rand() < density:
                new_sol[ue] = self.rng.randint(0, self.env.num_bs)
        return new_sol
