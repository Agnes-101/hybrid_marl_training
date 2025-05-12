import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class FLAOptimization:
    def __init__(self, env: NetworkEnvironment, iterations=20, kpi_logger=None):
        """Fick's Law Algorithm for 6G user association optimization"""
        self.env = env
        self.particles = 30    # Population size
        self.iterations = iterations
        self.diffusion_rate = 0.8     # Initial diffusion coefficient
        self.random_walk_prob = 0.3   # Exploration probability
        self.time_step = 0.5          # Virtual time step for diffusion
        self.decay_factor = 0.95      # Parameter decay rate
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
        
        # Initialize population
        solutions = np.array([self._random_solution(num_ue, num_bs) 
                            for _ in range(self.particles)])
        best_fitness = -np.inf
        self.best_solution = solutions[0].copy()

        for iteration in range(self.iterations):
            # Adaptive parameter adjustment
            current_diffusion = self.diffusion_rate * (self.decay_factor ** iteration)
            current_walk = self.random_walk_prob * (1 - iteration/self.iterations)
            
            # Update solutions using FLA dynamics
            solutions = self._update_particles(solutions, current_diffusion, current_walk)
            
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
                    algorithm="fla",
                    metrics=current_metrics
                )
                print(f"FLA Iter {iteration}: Best Fitness = {best_fitness:.4f}")

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
                "algorithm": "fla"
            }
        }

    def _random_solution(self, num_ue: int, num_bs: int) -> np.ndarray:
        """Generate random UE-BS associations"""
        return self.rng.randint(0, num_bs, size=num_ue)

    def _update_particles(self, solutions: np.ndarray, diffusion: float, walk_prob: float) -> np.ndarray:
        """Update solutions using Fick's Law dynamics"""
        new_solutions = []
        num_bs = self.env.num_bs
        
        for sol in solutions:
            # Calculate concentration gradient (fitness difference)
            current_fitness = self.env.evaluate_detailed_solution(sol)["fitness"]
            best_fitness = self.env.evaluate_detailed_solution(self.best_solution)["fitness"]
            gradient = best_fitness - current_fitness
            
            # Apply Fick's Law diffusion
            diffused_sol = self._diffusion_phase(sol, gradient, diffusion)
            
            # Apply random walk exploration
            final_sol = self._random_walk(diffused_sol, walk_prob)
            
            new_solutions.append(final_sol)
            
        return np.array(new_solutions)

    def _diffusion_phase(self, current_sol: np.ndarray, gradient: float, rate: float) -> np.ndarray:
        """Diffusion based on fitness gradient"""
        new_sol = current_sol.copy()
        num_ue = len(new_sol)
        
        # Calculate diffusion probability for each UE
        diffusion_prob = rate * (1 - np.exp(-self.time_step * abs(gradient)))
        
        for ue in range(num_ue):
            if self.rng.rand() < diffusion_prob:
                # Move toward best solution's association
                new_sol[ue] = self.best_solution[ue]
        return new_sol

    def _random_walk(self, current_sol: np.ndarray, prob: float) -> np.ndarray:
        """Brownian motion-inspired exploration"""
        new_sol = current_sol.copy()
        num_ue = len(new_sol)
        
        for ue in range(num_ue):
            if self.rng.rand() < prob:
                # Random walk with Gaussian step
                step = int(self.rng.normal(0, self.env.num_bs/4))
                new_sol[ue] = (current_sol[ue] + step) % self.env.num_bs
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