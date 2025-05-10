import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class RainbowOptimization:
    def __init__(self, env: NetworkEnvironment, kpi_logger=None):
        """Rainbow Optimization Algorithm for 6G user association"""
        self.env = env
        self.rays = 30          # Population size
        self.iterations = 20
        self.refraction_rate = 0.7   # Light refraction probability
        self.dispersion_factor = 0.4 # Wavelength dispersion strength
        self.spectrum_bands = 7      # Number of color bands
        self.prism_effect = 1.2      # Exploitation intensifier
        self.seed = 42
        self.kpi_logger = kpi_logger
        
        # Visualization and tracking
        self.positions = np.empty((0, 2))  # (mean_assoc, std_assoc)
        self.fitness_history = []
        self.rng = np.random.RandomState(self.seed)
        self.best_solution = None
        self.spectrum_weights = np.linspace(0.1, 1.0, self.spectrum_bands)  # Color weights

    def run(self, visualize_callback=None, kpi_logger=None) -> dict:
        """Main optimization process"""
        original_state = self.env.get_state_snapshot()
        num_ue = self.env.num_ue
        num_bs = self.env.num_bs
        
        # Initialize rainbow rays
        solutions = np.array([self._random_solution(num_ue, num_bs) 
                            for _ in range(self.rays)])
        best_fitness = -np.inf
        self.best_solution = solutions[0].copy()

        for iteration in range(self.iterations):
            # Adaptive parameter adjustment
            current_refraction = self.refraction_rate * (1 - iteration/self.iterations)
            current_prism = self.prism_effect * (iteration/self.iterations)
            
            # Update solutions using light phenomena
            solutions = self._refract_rays(solutions, current_refraction, current_prism)
            
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
                    algorithm="rainbow",
                    metrics=current_metrics
                )
                print(f"Rainbow Iter {iteration}: Best Fitness = {best_fitness:.4f}")

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
                "algorithm": "rainbow"
            }
        }

    def _random_solution(self, num_ue: int, num_bs: int) -> np.ndarray:
        """Generate random UE-BS associations"""
        return self.rng.randint(0, num_bs, size=num_ue)

    def _refract_rays(self, solutions: np.ndarray, refraction: float, prism: float) -> np.ndarray:
        """Update solutions using light refraction and dispersion"""
        new_solutions = []
        num_bs = self.env.num_bs
        
        for sol in solutions:
            # Select color band strategy
            color_idx = self.rng.randint(0, self.spectrum_bands)
            color_weight = self.spectrum_weights[color_idx]
            
            if self.rng.rand() < refraction:
                # Refraction phase: wavelength-based exploration
                new_sol = self._wavelength_refraction(sol, color_weight)
            else:
                # Prism phase: focused exploitation
                new_sol = self._prism_exploitation(sol, prism)
                
            new_solutions.append(new_sol)
            
        return np.array(new_solutions)

    def _wavelength_refraction(self, current_sol: np.ndarray, wavelength: float) -> np.ndarray:
        """Wavelength-dependent exploration with dispersion"""
        new_sol = current_sol.copy()
        num_ue = len(new_sol)
        
        for ue in range(num_ue):
            if self.rng.rand() < self.dispersion_factor:
                # Dispersion-based random walk
                step = int(wavelength * self.rng.normal(0, self.env.num_bs/2))
                new_sol[ue] = (current_sol[ue] + step) % self.env.num_bs
        return new_sol

    def _prism_exploitation(self, current_sol: np.ndarray, prism: float) -> np.ndarray:
        """Focused exploitation using prism effect"""
        new_sol = current_sol.copy()
        best_sol = self.best_solution
        num_ue = len(new_sol)
        
        # Calculate intensity gradient
        current_fitness = self.env.evaluate_detailed_solution(current_sol)["fitness"]
        best_fitness = self.env.evaluate_detailed_solution(best_sol)["fitness"]
        intensity = (best_fitness - current_fitness) / (best_fitness + 1e-10)
        
        for ue in range(num_ue):
            if self.rng.rand() < prism * intensity:
                # Align with best solution's associations
                new_sol[ue] = best_sol[ue]
            else:
                # Local refinement
                new_sol[ue] = (current_sol[ue] + self.rng.randint(-1, 2)) % self.env.num_bs
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