import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class DandelionOptimization:
    def __init__(self, env: NetworkEnvironment,iterations=20,seeds = 30,wind_factor = 0.5,lift_coeff = 1.2,
                descent_rate = 0.9,  kpi_logger=None):
        """Dandelion Optimizer for 6G user association"""
        self.env = env
        self.seeds = seeds       # Population size
        self.iterations = iterations
        self.wind_factor = wind_factor  # Controls exploration magnitude
        self.lift_coeff = lift_coeff   # Initial ascent coefficient
        self.descent_rate = descent_rate # Descent coefficient per iteration
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
        
        # Initialize dandelion seeds (solutions)
        solutions = np.array([self._random_solution(num_ue, num_bs) 
                            for _ in range(self.seeds)])
        best_fitness = -np.inf
        self.best_solution = solutions[0].copy()

        for iteration in range(self.iterations):
            # Adaptive parameters
            current_lift = self.lift_coeff * (self.descent_rate ** iteration)
            wind_strength = self.wind_factor * (1 - iteration/self.iterations)
            
            # Update seed positions
            solutions = self._update_seeds(solutions, wind_strength, current_lift)
            
            # Evaluate fitness
            fitness_values = np.array([self.env.evaluate_detailed_solution(sol)["fitness"] 
                                    for sol in solutions])
            current_best_idx = np.argmax(fitness_values)
            
            # Update global best
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
                    algorithm="do",
                    metrics=current_metrics
                )
                print(f"DO Iter {iteration}: Best Fitness = {best_fitness:.4f}")

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
                "algorithm": "do"
            }
        }

    def _random_solution(self, num_ue: int, num_bs: int) -> np.ndarray:
        """Generate random UE-BS associations"""
        return self.rng.randint(0, num_bs, size=num_ue)

    def _update_seeds(self, solutions: np.ndarray, wind_strength: float, lift: float) -> np.ndarray:
        """Update seed positions based on dandelion flight dynamics"""
        new_solutions = []
        num_bs = self.env.num_bs
        
        for sol in solutions:
            # Calculate movement vector
            movement = np.zeros_like(sol)
            for ue in range(len(sol)):
                # Wind-driven exploration with lift effect
                if self.rng.rand() < wind_strength:
                    movement[ue] = int(lift * self.rng.normal(0, num_bs/2))
                else:
                    # Local exploitation with reduced lift
                    movement[ue] = int(0.2 * lift * self.rng.normal(0, num_bs/4))
                    
            # Apply movement with boundary control
            new_sol = np.clip(sol + movement, 0, num_bs-1).astype(int)
            new_solutions.append(new_sol)
            
        return np.array(new_solutions)

    def _update_visualization(self, iteration: int):
        """Track solution diversity metrics"""
        current_solutions = self.env.get_current_associations()
        mean_assoc = np.mean(current_solutions)
        std_assoc = np.std(current_solutions)
        
        self.positions = np.vstack([self.positions, [mean_assoc, std_assoc]])
        self.fitness_history.append(
            self.env.evaluate_detailed_solution(self.best_solution)["fitness"]
        )