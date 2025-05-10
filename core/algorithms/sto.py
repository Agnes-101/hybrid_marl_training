import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class STOOptimization:
    def __init__(self, env: NetworkEnvironment, kpi_logger=None):
        """Siberian Tiger Optimizer for 6G user association and load balancing"""
        self.env = env
        self.tigers = 30       # Population size
        self.iterations = 20
        self.territory_radius = 0.4  # Initial exploration range
        self.attack_intensity = 1.5   # Exploitation strength
        self.marking_rate = 0.2       # Territory marking frequency
        self.seed = 42
        self.kpi_logger = kpi_logger
        
        # Tracking and visualization
        self.positions = np.empty((0, 2))  # (mean_assoc, std_assoc)
        self.fitness_history = []
        self.rng = np.random.RandomState(self.seed)
        self.best_solution = None
        self.territory_center = None

    def run(self, visualize_callback=None, kpi_logger=None) -> dict:
        """Main optimization process"""
        original_state = self.env.get_state_snapshot()
        num_ue = self.env.num_ue
        num_bs = self.env.num_bs
        
        # Initialize tiger population and territory
        solutions = np.array([self._random_solution(num_ue, num_bs) 
                            for _ in range(self.tigers)])
        self.territory_center = solutions[0].copy()
        best_fitness = -np.inf
        self.best_solution = solutions[0].copy()

        for iteration in range(self.iterations):
            # Adaptive parameter adjustment
            current_radius = self.territory_radius * (1 - iteration/self.iterations)
            attack_strength = self.attack_intensity * (iteration/self.iterations)
            
            # Update tiger positions
            solutions = self._hunt(solutions, current_radius, attack_strength)
            
            # Update territory markings
            self._mark_territory(solutions)
            
            # Evaluate fitness
            fitness_values = np.array([self.env.evaluate_detailed_solution(sol)["fitness"] 
                                    for sol in solutions])
            current_best_idx = np.argmax(fitness_values)
            
            # Update alpha tiger
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
                    algorithm="sto",
                    metrics=current_metrics
                )
                print(f"STO Iter {iteration}: Best Fitness = {best_fitness:.4f}")

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
                "algorithm": "sto"
            }
        }

    def _random_solution(self, num_ue: int, num_bs: int) -> np.ndarray:
        """Generate random UE-BS associations"""
        return self.rng.randint(0, num_bs, size=num_ue)

    def _hunt(self, solutions: np.ndarray, radius: float, attack: float) -> np.ndarray:
        """Tiger hunting behavior with territorial awareness"""
        new_solutions = []
        num_bs = self.env.num_bs
        
        for sol in solutions:
            # Choose hunting strategy
            if self.rng.rand() < attack:
                new_sol = self._sneak_attack(sol, self.best_solution)
            else:
                new_sol = self._patrol_territory(sol, radius)
            new_solutions.append(new_sol)
            
        return np.array(new_solutions)

    def _sneak_attack(self, current_sol: np.ndarray, alpha_sol: np.ndarray) -> np.ndarray:
        """Focused attack on prey (best solution)"""
        new_sol = current_sol.copy()
        num_ue = len(new_sol)
        
        for ue in range(num_ue):
            if current_sol[ue] != alpha_sol[ue]:
                # Calculate attack direction
                direction = 1 if alpha_sol[ue] > current_sol[ue] else -1
                step_size = int(self.rng.exponential(1) + 1)
                new_bs = (current_sol[ue] + direction * step_size) % self.env.num_bs
                new_sol[ue] = new_bs
        return new_sol

    def _patrol_territory(self, current_sol: np.ndarray, radius: float) -> np.ndarray:
        """Territory exploration with scent marking"""
        new_sol = current_sol.copy()
        num_ue = len(new_sol)
        
        for ue in range(num_ue):
            if self.rng.rand() < self.marking_rate:
                # Follow territory scent
                new_sol[ue] = self.territory_center[ue]
            else:
                # Random patrol within territory radius
                max_step = int(radius * self.env.num_bs)
                step = self.rng.randint(-max_step, max_step+1)
                new_sol[ue] = (current_sol[ue] + step) % self.env.num_bs
        return new_sol

    def _mark_territory(self, solutions: np.ndarray):
        """Update territory center based on population distribution"""
        # Calculate weighted center from all solutions
        weighted_solutions = np.array([sol * (i+1) for i, sol in enumerate(solutions)])
        self.territory_center = np.mean(weighted_solutions, axis=0, dtype=int) % self.env.num_bs

    def _update_visualization(self, iteration: int):
        """Track territory metrics for visualization"""
        current_solutions = self.env.get_current_associations()
        mean_assoc = np.mean(current_solutions)
        std_assoc = np.std(current_solutions)
        
        self.positions = np.vstack([self.positions, [mean_assoc, std_assoc]])
        self.fitness_history.append(
            self.env.evaluate_detailed_solution(self.best_solution)["fitness"]
        )