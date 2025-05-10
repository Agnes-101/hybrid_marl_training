import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class CheetahOptimization:
    def __init__(self, env: NetworkEnvironment, kpi_logger=None):
        """Cheetah Optimizer for 6G user association and load balancing"""
        self.env = env
        self.cheetahs = 30     # Population size
        self.iterations = 20
        self.sprint_prob = 0.6     # Probability of high-speed chase
        self.rest_threshold = 0.3  # Energy conservation threshold
        self.acceleration = 1.2    # Sprint speed multiplier
        self.energy_decay = 0.95   # Energy depletion rate
        self.seed = 42
        self.kpi_logger = kpi_logger
        
        # Visualization and tracking
        self.positions = np.empty((0, 2))  # (mean_assoc, std_assoc)
        self.fitness_history = []
        self.rng = np.random.RandomState(self.seed)
        self.best_solution = None
        self.energy_levels = np.ones(self.cheetahs)  # Individual energy reserves

    def run(self, visualize_callback=None, kpi_logger=None) -> dict:
        """Main optimization process"""
        original_state = self.env.get_state_snapshot()
        num_ue = self.env.num_ue
        num_bs = self.env.num_bs
        
        # Initialize cheetah population
        solutions = np.array([self._random_solution(num_ue, num_bs) 
                            for _ in range(self.cheetahs)])
        best_fitness = -np.inf
        self.best_solution = solutions[0].copy()

        for iteration in range(self.iterations):
            # Update energy levels
            self.energy_levels *= self.energy_decay
            
            # Adaptive parameter adjustment
            current_sprint_prob = self.sprint_prob * (self.energy_levels.mean())
            current_acceleration = self.acceleration * (iteration/self.iterations)
            
            # Update positions
            solutions = self._hunt(solutions, current_sprint_prob, current_acceleration)
            
            # Evaluate fitness
            fitness_values = np.array([self.env.evaluate_detailed_solution(sol)["fitness"] 
                                    for sol in solutions])
            current_best_idx = np.argmax(fitness_values)
            
            # Update best solution and energy
            if fitness_values[current_best_idx] > best_fitness:
                best_fitness = fitness_values[current_best_idx]
                self.best_solution = solutions[current_best_idx].copy()
                self.energy_levels[current_best_idx] = 1.0  # Reward successful hunt

            # Logging and visualization
            # self._update_visualization(iteration)
            current_metrics = self.env.evaluate_detailed_solution(self.best_solution)
            
            if self.kpi_logger:
                self.kpi_logger.log_metrics(
                    episode=iteration,
                    phase="metaheuristic",
                    algorithm="cheetah",
                    metrics=current_metrics
                )
                print(f"Cheetah Iter {iteration}: Best Fitness = {best_fitness:.4f}")

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
                "algorithm": "cheetah"
            }
        }

    def _random_solution(self, num_ue: int, num_bs: int) -> np.ndarray:
        """Generate random UE-BS associations"""
        return self.rng.randint(0, num_bs, size=num_ue)

    def _hunt(self, solutions: np.ndarray, sprint_prob: float, acceleration: float) -> np.ndarray:
        """Cheetah hunting behavior with sprint-rest cycles"""
        new_solutions = []
        num_bs = self.env.num_bs
        
        for idx, sol in enumerate(solutions):
            if self.energy_levels[idx] > self.rest_threshold and self.rng.rand() < sprint_prob:
                # Sprint phase: high-speed chase
                new_sol = self._sprint(sol, acceleration)
                self.energy_levels[idx] *= 0.8  # Energy depletion
            else:
                # Rest phase: local observation
                new_sol = self._rest(sol)
                self.energy_levels[idx] = min(1.0, self.energy_levels[idx] + 0.1)
                
            new_solutions.append(new_sol)
            
        return np.array(new_solutions)

    def _sprint(self, current_sol: np.ndarray, acceleration: float) -> np.ndarray:
        """High-speed chase toward prey (best solution)"""
        new_sol = current_sol.copy()
        num_ue = len(new_sol)
        
        # Calculate direction to best solution
        direction = self.best_solution - current_sol
        
        # Apply accelerated movement with momentum
        for ue in range(num_ue):
            step = int(acceleration * direction[ue] + self.rng.normal(0, 1))
            new_sol[ue] = (current_sol[ue] + step) % self.env.num_bs
            
        return new_sol

    def _rest(self, current_sol: np.ndarray) -> np.ndarray:
        """Energy conservation through local refinement"""
        new_sol = current_sol.copy()
        num_ue = len(new_sol)
        
        # Small random adjustments
        for ue in range(num_ue):
            if self.rng.rand() < 0.3:  # 30% chance to adjust
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