import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class HBAOptimization:
    def __init__(self, env: NetworkEnvironment,iterations=20,badgers = 30,intensity = 0.9,density_factor = 1.0,
                honey_prob = 0.5,kpi_logger=None):
        """Honey Badger Algorithm for 6G user association optimization"""
        self.env = env
        self.badgers = badgers      # Population size
        self.iterations = iterations
        self.intensity = intensity  # Smell intensity coefficient
        self.density_factor = density_factor  # Exploration density
        self.honey_prob = honey_prob  # Probability of honey phase
        self.seed = 42
        self.kpi_logger = kpi_logger
        
        # Visualization and tracking
        self.positions = np.empty((0, 2))  # (mean_assoc, std_assoc)
        self.fitness_history = []
        self.rng = np.random.RandomState(self.seed)
        self.best_solution = None

    def run(self, visualize_callback=None, kpi_logger=None) -> dict:
        """Main optimization process"""
        original_state = self.env.get_state_snapshot()
        num_ue = self.env.num_ue
        num_bs = self.env.num_bs
        
        # Initialize badger population
        solutions = np.array([self._random_solution(num_ue, num_bs) 
                            for _ in range(self.badgers)])
        best_fitness = -np.inf
        self.best_solution = solutions[0].copy()

        for iteration in range(self.iterations):
            # Adaptive parameter adjustment
            decay_factor = 1 - (iteration / self.iterations)
            current_density = self.density_factor * decay_factor
            
            # Update badger positions
            solutions = self._update_positions(solutions, current_density)
            
            # Evaluate fitness
            fitness_values = np.array([self.env.evaluate_detailed_solution(sol)["fitness"] 
                                    for sol in solutions])
            current_best_idx = np.argmax(fitness_values)
            
            # Update best solution
            if fitness_values[current_best_idx] > best_fitness:
                best_fitness = fitness_values[current_best_idx]
                self.best_solution = solutions[current_best_idx].copy()

            # Logging and visualization
            # self._update_visualization(iteration)
            current_metrics = self.env.evaluate_detailed_solution(self.best_solution)
            
            if self.kpi_logger:
                self.kpi_logger.log_metrics(
                    episode=iteration,
                    phase="metaheuristic",
                    algorithm="hba",
                    metrics=current_metrics
                )
                print(f"HBA Iter {iteration}: Best Fitness = {best_fitness:.4f}")

            if visualize_callback:
                viz_metrics = {
                    "fitness": current_metrics["fitness"],
                    "average_sinr": current_metrics["average_sinr"],
                    "fairness": current_metrics["fairness"]
                }
                visualize_callback(viz_metrics, self.best_solution)

        # Finalize and return results
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

    def _random_solution(self, num_ue: int, num_bs: int) -> np.ndarray:
        """Generate random UE-BS associations"""
        return self.rng.randint(0, num_bs, size=num_ue)

    def _update_positions(self, solutions: np.ndarray, density: float) -> np.ndarray:
        """Update positions using digging and honey phases"""
        new_solutions = []
        num_bs = self.env.num_bs
        fitness_values = np.array([self.env.evaluate_detailed_solution(sol)["fitness"] 
                                 for sol in solutions])
        
        # Normalize fitness values for intensity calculation
        max_fitness = np.max(fitness_values)
        min_fitness = np.min(fitness_values)
        normalized_fitness = (fitness_values - min_fitness) / (max_fitness - min_fitness + 1e-10)
        
        for idx, sol in enumerate(solutions):
            if self.rng.rand() < self.honey_prob:
                # Honey phase: move toward best solution
                new_sol = self._honey_phase(sol, density)
            else:
                # Digging phase: local exploration
                new_sol = self._digging_phase(sol, normalized_fitness[idx], density)
            
            new_solutions.append(new_sol)
            
        return np.array(new_solutions)

    def _digging_phase(self, current_sol: np.ndarray, intensity: float, density: float) -> np.ndarray:
        """Exploration phase with smell intensity guidance"""
        new_sol = current_sol.copy()
        num_ue = len(new_sol)
        
        for ue in range(num_ue):
            if self.rng.rand() < density:
                # Random exploration weighted by intensity
                delta = int(self.intensity * density * self.rng.normal(0, self.env.num_bs/2))
                new_bs = (current_sol[ue] + delta) % self.env.num_bs
                new_sol[ue] = new_bs
        return new_sol

    def _honey_phase(self, current_sol: np.ndarray, density: float) -> np.ndarray:
        """Exploitation phase moving toward best solution"""
        new_sol = current_sol.copy()
        best_sol = self.best_solution
        
        for ue in range(len(new_sol)):
            if self.rng.rand() < self.intensity:
                # Guided movement toward best solution
                if current_sol[ue] != best_sol[ue]:
                    step = 1 if best_sol[ue] > current_sol[ue] else -1
                    new_sol[ue] = (current_sol[ue] + step) % self.env.num_bs
                else:
                    # Local refinement
                    new_sol[ue] = (current_sol[ue] + self.rng.randint(-1, 2)) % self.env.num_bs
            elif self.rng.rand() < density:
                # Random component
                new_sol[ue] = self.rng.randint(0, self.env.num_bs)
                
        return new_sol

    def _update_visualization(self, iteration: int):
        """Track solution diversity metrics"""
        current_solutions = self.env.get_current_associations()
        mean_assoc = np.mean(current_solutions)
        std_assoc = np.std(current_solutions)
        
        self.positions = np.vstack([self.positions, [mean_assoc, std_assoc]])
        self.fitness_history.append(
            self.env.evaluate_detailed_solution(self.best_solution)["fitness"]
        )