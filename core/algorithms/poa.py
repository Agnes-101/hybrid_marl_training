import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class PelicanOptimization:
    def __init__(self, env: NetworkEnvironment,iterations=20, kpi_logger=None):
        """Pelican Optimization Algorithm for 6G user association"""
        self.env = env
        self.pelicans = 30     # Population size
        self.iterations = iterations
        self.initial_movement = 0.8    # Exploration rate
        self.scoop_intensity = 0.2     # Exploitation rate
        self.decay_factor = 0.95       # Exploration decay rate
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
        
        # Initialize pelican population
        solutions = np.array([self._random_solution(num_ue, num_bs) 
                            for _ in range(self.pelicans)])
        best_fitness = -np.inf
        self.best_solution = solutions[0].copy()

        for iteration in range(self.iterations):
            # Adaptive parameter adjustment
            movement_rate = self.initial_movement * (self.decay_factor ** iteration)
            scoop_rate = self.scoop_intensity * (1 + iteration/self.iterations)
            
            # Update pelican positions
            solutions = self._hunting_behavior(solutions, movement_rate, scoop_rate)
            
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
                    algorithm="poa",
                    metrics=current_metrics
                )
                print(f"POA Iter {iteration}: Best Fitness = {best_fitness:.4f}")

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
                "algorithm": "poa"
            }
        }

    def _random_solution(self, num_ue: int, num_bs: int) -> np.ndarray:
        """Generate random UE-BS associations"""
        return self.rng.randint(0, num_bs, size=num_ue)

    def _hunting_behavior(self, solutions: np.ndarray, movement: float, scoop: float) -> np.ndarray:
        """Implement pelican's two-phase hunting strategy"""
        new_solutions = []
        num_bs = self.env.num_bs
        
        for sol in solutions:
            # Phase 1: Diving toward prey (exploration)
            exploratory_sol = self._dive_toward_prey(sol, movement)
            
            # Phase 2: Scooping movement (exploitation)
            exploitative_sol = self._scoop_movement(exploratory_sol, scoop)
            
            new_solutions.append(exploitative_sol)
            
        return np.array(new_solutions)

    def _dive_toward_prey(self, current_sol: np.ndarray, rate: float) -> np.ndarray:
        """Exploration phase: move toward best solution"""
        new_sol = current_sol.copy()
        for ue in range(len(new_sol)):
            if self.rng.rand() < rate:
                new_sol[ue] = self.best_solution[ue]
        return new_sol

    def _scoop_movement(self, current_sol: np.ndarray, intensity: float) -> np.ndarray:
        """Exploitation phase: local refinement"""
        new_sol = current_sol.copy()
        num_ue = len(new_sol)
        num_bs = self.env.num_bs
        
        # Calculate number of UEs to modify based on intensity
        num_changes = max(1, int(intensity * num_ue))
        ues_to_change = self.rng.choice(num_ue, size=num_changes, replace=False)
        
        for ue in ues_to_change:
            # Perturb association with neighborhood bias
            if self.rng.rand() < 0.7:  # 70% chance for adjacent BS
                new_bs = (current_sol[ue] + self.rng.choice([-1, 1])) % num_bs
            else:  # 30% chance for random jump
                new_bs = self.rng.randint(0, num_bs)
            new_sol[ue] = new_bs
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