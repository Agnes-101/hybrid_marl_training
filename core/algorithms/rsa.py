import numpy as np
from core.envs.custom_channel_env import NetworkEnvironment

class RSAOptimization:
    def __init__(self, env: NetworkEnvironment, iterations=20, reptiles = 30, alpha = 0.1,beta = 1.5,hunting_prob = 0.7,kpi_logger=None):
        """Reptile Search Algorithm for 6G user association optimization"""
        self.env = env
        
        self.reptiles = reptiles     # Population size
        self.iterations = iterations
        self.alpha = alpha      # Exploration control
        self.beta = beta       # Exploitation intensity
        self.hunting_prob = hunting_prob  # Probability of hunting behavior
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
        
        # Initialize reptile population
        solutions = np.array([self._random_solution(num_ue, num_bs) 
                            for _ in range(self.reptiles)])
        best_fitness = -np.inf
        self.best_solution = solutions[0].copy()

        for iteration in range(self.iterations):
            # Adaptive parameter adjustment
            exploration_rate = self.alpha * (1 - iteration/self.iterations)
            exploitation_power = self.beta * (iteration/self.iterations)
            
            # Update reptile positions
            solutions = self._update_reptiles(solutions, exploration_rate, exploitation_power)
            
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
                    algorithm="rsa",
                    metrics=current_metrics
                )
                print(f"RSA Iter {iteration}: Best Fitness = {best_fitness:.4f}")

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
                "algorithm": "rsa"
            }
        }

    def _random_solution(self, num_ue: int, num_bs: int) -> np.ndarray:
        """Generate random UE-BS associations"""
        return self.rng.randint(0, num_bs, size=num_ue)

    def _update_reptiles(self, solutions: np.ndarray, exploration: float, exploitation: float) -> np.ndarray:
        """Update positions based on crocodile hunting strategies"""
        new_solutions = []
        num_bs = self.env.num_bs
        
        for sol in solutions:
            if self.rng.rand() < self.hunting_prob:
                # Hunting phase: coordinated attack
                new_sol = self._coordinated_attack(sol, exploitation)
            else:
                # Encircling phase: exploration
                new_sol = self._encircle_prey(sol, exploration)
            new_solutions.append(new_sol)
            
        return np.array(new_solutions)

    def _coordinated_attack(self, current_sol: np.ndarray, exploitation: float) -> np.ndarray:
        """Exploitation phase: focused attack on best solution"""
        new_sol = current_sol.copy()
        best_sol = self.best_solution
        
        for ue in range(len(new_sol)):
            if self.rng.rand() < exploitation:
                # Direct alignment with best solution
                new_sol[ue] = best_sol[ue]
            else:
                # Local refinement with decreasing randomness
                delta = int(self.rng.normal(0, self.num_bs * (1 - exploitation)))
                new_sol[ue] = (current_sol[ue] + delta) % self.num_bs
        return new_sol

    def _encircle_prey(self, current_sol: np.ndarray, exploration: float) -> np.ndarray:
        """Exploration phase: wide search around current position"""
        new_sol = current_sol.copy()
        num_ue = len(new_sol)
        
        for ue in range(num_ue):
            if self.rng.rand() < exploration:
                # Large random jumps
                new_sol[ue] = self.rng.randint(0, self.env.num_bs)
            else:
                # Neighborhood search
                new_sol[ue] = (current_sol[ue] + self.rng.randint(-2, 3)) % self.env.num_bs
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