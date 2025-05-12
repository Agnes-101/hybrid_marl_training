import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class GTOOptimization:
    def __init__(self, env: NetworkEnvironment,iterations=20, kpi_logger=None):
        """Gorilla Troops Optimizer for user association in 6G networks"""
        self.env = env
        self.gorillas = 30    # Population size
        self.iterations = iterations
        self.silverback_influence = 0.8  # Leader's decision weight
        self.migration_prob = 0.3        # Exploration probability
        self.social_factor = 0.5         # Group interaction strength
        self.seed = 42
        self.kpi_logger = kpi_logger
        
        # Tracking and visualization
        self.positions = np.empty((0, 2))  # (mean_assoc, std_assoc)
        self.fitness_history = []
        self.rng = np.random.RandomState(self.seed)
        self.best_solution = None

    def run(self, visualize_callback=None, kpi_logger=None) -> dict:
        """Main optimization process"""
        original_state = self.env.get_state_snapshot()
        num_ue = self.env.num_ue
        num_bs = self.env.num_bs
        
        # Initialize gorilla population
        solutions = np.array([self._random_solution(num_ue, num_bs) 
                            for _ in range(self.gorillas)])
        best_fitness = -np.inf
        self.best_solution = solutions[0].copy()

        for iteration in range(self.iterations):
            # Adaptive parameter adjustment
            migration_rate = self.migration_prob * (1 - iteration/self.iterations)
            leadership_strength = self.silverback_influence * (iteration/self.iterations)
            
            # Update gorilla positions
            solutions = self._update_troops(solutions, migration_rate, leadership_strength)
            
            # Evaluate fitness
            fitness_values = np.array([self.env.evaluate_detailed_solution(sol)["fitness"] 
                                    for sol in solutions])
            current_best_idx = np.argmax(fitness_values)
            
            # Update silverback (best solution)
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
                    algorithm="gto",
                    metrics=current_metrics
                )
                print(f"GTO Iter {iteration}: Best Fitness = {best_fitness:.4f}")

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
                "algorithm": "gto"
            }
        }

    def _random_solution(self, num_ue: int, num_bs: int) -> np.ndarray:
        """Generate random UE-BS associations"""
        return self.rng.randint(0, num_bs, size=num_ue)

    def _update_troops(self, solutions: np.ndarray, migration_rate: float, leadership: float) -> np.ndarray:
        """Update gorilla positions based on troop behavior"""
        new_solutions = []
        num_bs = self.env.num_bs
        
        for idx, sol in enumerate(solutions):
            if self.rng.rand() < migration_rate:
                # Migration to unknown place (exploration)
                new_sol = self._migrate(sol)
            else:
                # Follow silverback/group (exploitation)
                new_sol = self._follow_leader(sol, leadership)
            new_solutions.append(new_sol)
            
        return np.array(new_solutions)

    def _migrate(self, current_sol: np.ndarray) -> np.ndarray:
        """Exploration phase: random migration with social learning"""
        new_sol = current_sol.copy()
        num_ue = len(new_sol)
        
        # Randomly change associations based on social factor
        for ue in range(num_ue):
            if self.rng.rand() < self.social_factor:
                new_sol[ue] = self.rng.randint(0, self.env.num_bs)
        return new_sol

    def _follow_leader(self, current_sol: np.ndarray, leadership: float) -> np.ndarray:
        """Exploitation phase: follow silverback with some randomness"""
        new_sol = current_sol.copy()
        silverback_sol = self.best_solution
        
        # Blend current solution with leader's solution
        for ue in range(len(new_sol)):
            if self.rng.rand() < leadership:
                new_sol[ue] = silverback_sol[ue]
            else:
                # Small random adjustments around current position
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