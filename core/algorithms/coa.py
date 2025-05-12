import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class COAOptimization:
    def __init__(self, env: NetworkEnvironment, iterations=20,kpi_logger=None):
        """Coati Optimization Algorithm for 6G network optimization"""
        self.env = env
        self.coatis = 30  # Population size
        self.iterations = iterations
        self.attack_rate = 0.1  # Probability of attacking prey
        self.explore_rate = 0.7  # Exploration probability
        self.seed = 42
        self.kpi_logger = kpi_logger
        
        # Visualization and tracking
        self.positions = np.empty((0, 2))
        self.fitness_history = []
        self.rng = np.random.RandomState(self.seed)
        self.best_solution = None

    def run(self, visualize_callback=None, kpi_logger=None) -> dict:
        """Main optimization process"""
        original_state = self.env.get_state_snapshot()
        num_ue = self.env.num_ue
        num_bs = self.env.num_bs
        
        # Initialize coatis' positions (solutions)
        solutions = np.array([self._random_solution(num_ue, num_bs) for _ in range(self.coatis)])
        best_fitness = -np.inf
        self.best_solution = solutions[0].copy()

        for iteration in range(self.iterations):
            # Adaptive parameter adjustment
            exploration = self.explore_rate * (1 - iteration/self.iterations)
            
            # Update solutions using COA strategies
            solutions = self._update_positions(solutions, self.best_solution, exploration)
            
            # Evaluate fitness
            fitness_values = np.array([self.env.evaluate_detailed_solution(sol)["fitness"] for sol in solutions])
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
                    algorithm="coa",
                    metrics=current_metrics
                )
                print(f"COA Iter {iteration}: Best Fitness = {best_fitness:.4f}")

            # if visualize_callback:
            #     viz_metrics = {
            #         "fitness": current_metrics["fitness"],
            #         "average_sinr": current_metrics["average_sinr"],
            #         "fairness": current_metrics["fairness"]
            #     }
            #     visualize_callback(viz_metrics, self.best_solution)

        # Finalize and return results
        self.env.set_state_snapshot(original_state)
        self.env.apply_solution(self.best_solution)
        
        return {
            "solution": self.best_solution,
            "metrics": self.env.evaluate_detailed_solution(self.best_solution),
            "agents": {
                "positions": self.positions.tolist(),
                "fitness": self.fitness_history,
                "algorithm": "coa"
            }
        }

    def _random_solution(self, num_ue: int, num_bs: int) -> np.ndarray:
        """Generate random association matrix"""
        return self.rng.randint(0, num_bs, size=num_ue)

    def _update_positions(self, solutions: np.ndarray, best_solution: np.ndarray, exploration: float) -> np.ndarray:
        """Update coatis' positions based on hunting behavior"""
        new_solutions = []
        for sol in solutions:
            if self.rng.rand() < self.attack_rate:
                # Attack phase: move toward best solution
                new_sol = self._attack_move(sol, best_solution)
            else:
                # Exploration phase
                new_sol = self._explore_move(sol, exploration)
            new_solutions.append(new_sol)
        return np.array(new_solutions)

    def _attack_move(self, current_sol: np.ndarray, best_sol: np.ndarray) -> np.ndarray:
        """Move toward prey (best solution) with stochastic elements"""
        new_sol = current_sol.copy()
        mask = self.rng.rand(len(new_sol)) < 0.5  # 50% chance to follow best
        new_sol[mask] = best_sol[mask]
        return new_sol

    def _explore_move(self, current_sol: np.ndarray, exploration: float) -> np.ndarray:
        """Random exploration with directional bias"""
        new_sol = current_sol.copy()
        num_bs = self.env.num_bs
        
        # Random changes with increasing preference for better BSs
        for ue in range(len(new_sol)):
            if self.rng.rand() < exploration:
                new_sol[ue] = self.rng.randint(0, num_bs)
            else:
                # Slight bias toward current best BS for this UE
                current_bs = new_sol[ue]
                new_sol[ue] = (current_bs + self.rng.randint(0, num_bs)) % num_bs
        return new_sol

    def _update_visualization(self, iteration: int):
        """Track solution statistics for visualization"""
        # Calculate solution diversity metrics
        current_solutions = self.env.get_current_associations()
        mean_assoc = np.mean(current_solutions)
        std_assoc = np.std(current_solutions)
        
        self.positions = np.vstack([self.positions, [mean_assoc, std_assoc]])
        self.fitness_history.append(self.env.evaluate_detailed_solution(self.best_solution)["fitness"])