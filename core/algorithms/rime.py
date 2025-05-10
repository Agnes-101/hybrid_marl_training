import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class RIMEOptimization:
    def __init__(self, env: NetworkEnvironment, kpi_logger=None):
        """RIME Algorithm for 6G user association and load balancing"""
        self.env = env
        self.particles = 30    # Population size
        self.iterations = 20
        self.temperature = 1.0   # Initial phase control parameter
        self.cooling_rate = 0.95 # Temperature decay rate
        self.stability_threshold = 0.1 # Solution component freezing threshold
        self.phase_ratio = 0.6  # Soft/hard rime balance
        self.seed = 42
        self.kpi_logger = kpi_logger
        
        # Visualization and tracking
        self.positions = np.empty((0, 2))  # (mean_assoc, std_assoc)
        self.fitness_history = []
        self.rng = np.random.RandomState(self.seed)
        self.best_solution = None
        self.global_best_fitness = -np.inf

    def run(self, visualize_callback=None, kpi_logger=None) -> dict:
        """Main optimization process"""
        original_state = self.env.get_state_snapshot()
        num_ue = self.env.num_ue
        num_bs = self.env.num_bs
        
        # Initialize rime-ice particles
        solutions = np.array([self._random_solution(num_ue, num_bs) 
                            for _ in range(self.particles)])
        self.best_solution = solutions[0].copy()

        for iteration in range(self.iterations):
            # Update temperature and phase ratio
            self.temperature *= self.cooling_rate
            current_phase_ratio = self.phase_ratio * (1 - iteration/self.iterations)
            
            # Evaluate fitness
            fitness_values = np.array([self.env.evaluate_detailed_solution(sol)["fitness"] 
                                    for sol in solutions])
            
            # Update global best
            current_best_idx = np.argmax(fitness_values)
            if fitness_values[current_best_idx] > self.global_best_fitness:
                self.global_best_fitness = fitness_values[current_best_idx]
                self.best_solution = solutions[current_best_idx].copy()

            # Update solutions using RIME dynamics
            solutions = self._update_particles(solutions, fitness_values, current_phase_ratio)
            
            # Logging and visualization
            # self._update_visualization(iteration)
            current_metrics = self.env.evaluate_detailed_solution(self.best_solution)
            
            if self.kpi_logger:
                self.kpi_logger.log_metrics(
                    episode=iteration,
                    phase="metaheuristic",
                    algorithm="rime",
                    metrics=current_metrics
                )
                print(f"RIME Iter {iteration}: Best Fitness = {self.global_best_fitness:.4f}")

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
                "algorithm": "rime"
            }
        }

    def _random_solution(self, num_ue: int, num_bs: int) -> np.ndarray:
        """Generate random UE-BS associations"""
        return self.rng.randint(0, num_bs, size=num_ue)

    def _update_particles(self, solutions: np.ndarray, fitness_values: np.ndarray, phase_ratio: float) -> np.ndarray:
        """Update solutions using rime-ice formation dynamics"""
        new_solutions = []
        num_bs = self.env.num_bs
        normalized_fitness = (fitness_values - np.min(fitness_values)) / \
                           (np.max(fitness_values) - np.min(fitness_values) + 1e-10)
        
        for idx, sol in enumerate(solutions):
            # Determine phase composition
            if self.rng.rand() < phase_ratio:
                # Soft rime phase (exploration)
                new_sol = self._soft_rime_phase(sol, normalized_fitness[idx])
            else:
                # Hard rime phase (exploitation)
                new_sol = self._hard_rime_phase(sol)
            
            new_solutions.append(new_sol)
        
        return np.array(new_solutions)

    def _soft_rime_phase(self, current_sol: np.ndarray, fitness: float) -> np.ndarray:
        """Exploratory phase with dynamic freezing patterns"""
        new_sol = current_sol.copy()
        num_ue = len(new_sol)
        
        for ue in range(num_ue):
            # Probability of change decreases with fitness and temperature
            change_prob = (1 - fitness) * self.temperature
            
            if self.rng.rand() < change_prob:
                # Random walk with temperature-dependent step size
                step = int(self.rng.normal(0, self.env.num_bs * self.temperature))
                new_sol[ue] = (current_sol[ue] + step) % self.env.num_bs
        return new_sol

    def _hard_rime_phase(self, current_sol: np.ndarray) -> np.ndarray:
        """Exploitative phase with structural stabilization"""
        new_sol = current_sol.copy()
        num_ue = len(new_sol)
        best_sol = self.best_solution
        
        for ue in range(num_ue):
            # Freeze components that match global best
            if current_sol[ue] == best_sol[ue]:
                if self.rng.rand() < self.stability_threshold:
                    continue  # Maintain frozen component
                
            # Crystallization toward best solution
            delta = best_sol[ue] - current_sol[ue]
            step = np.sign(delta) * int(abs(delta) * (1 - self.temperature))
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