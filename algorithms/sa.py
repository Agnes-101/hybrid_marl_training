import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class SAOptimization:
    def __init__(self, env, kpi_logger=None):
        self.env=env
        self.num_users = env.num_ue
        self.num_cells = env.num_bs
        self.iterations = 50
        self.temperature = 100
        self.cooling_rate = 0.95
        self.seed = 42
        self.rng = np.random.RandomState(self.seed)
        # Initialize current solution using the seeded RNG
        self.current_solution = self.rng.randint(0, self.num_cells, size=self.num_users)
        self.kpi_logger = kpi_logger
        # Visualization states
        self.positions = np.empty((0, 3))  # (temperature, current_fit, best_fit)
        self.fitness_history = []
        
        # Initialize state
        self.current_solution = self.rng.randint(0, self.num_cells, self.num_users)
        self.best_solution = self.current_solution.copy()
        
    def fitness(self, solution):
        return self.env.evaluate_detailed_solution( solution)["fitness"]
    
    def run(self, env: NetworkEnvironment, visualize_callback: callable = None, kpi_logger=None) -> dict:        
        temperature = self.initial_temp
        current_fit = self.fitness(self.current_solution)
        best_fit = current_fit
        
        for iteration in range(self.iterations):
            # neighbor = self.current_solution.copy()
            # idx = self.rng.randint(0, self.num_users)
            # neighbor[idx] = self.rng.randint(0, self.num_cells)
            # neighbor_fit = self.fitness(neighbor)
            # delta = neighbor_fit - current_fit
            # if delta > 0 or self.rng.rand() < np.exp(delta / self.temperature):
            #     self.current_solution = neighbor
            #     current_fit = neighbor_fit
            #     if neighbor_fit > best_fit:
            #         best_solution = neighbor.copy()
            #         best_fit = neighbor_fit
            # self.temperature *= self.cooling_rate
            # Generate neighbor
            neighbor = self._generate_neighbor()
            neighbor_fit = self.fitness(neighbor)
            
            # Acceptance criteria
            delta = neighbor_fit - current_fit
            if delta > 0 or self.rng.rand() < np.exp(delta / temperature):
                self.current_solution = neighbor
                current_fit = neighbor_fit
                if neighbor_fit > best_fit:
                    self.best_solution = neighbor.copy()
                    best_fit = neighbor_fit
            
            # Cooling schedule
            temperature *= self.cooling_rate
            # ✅ DE-style logging
            current_metrics = env.evaluate_detailed_solution( self.best_solution)
            if self.kpi_logger:
                self.kpi_logger.log_metrics(
                    episode=iteration,
                    phase="metaheuristic",
                    algorithm="sa",
                    metrics=current_metrics
                )
            
            # ✅ Environment update
            self.env.apply_solution(self.best_solution)
            self.env.step({
                f"bs_{bs_id}": np.where(self.best_solution == bs_id)[0].tolist()
                for bs_id in range(self.env.num_bs)
            })

            # # ✅ Visualization updates
            # self._update_visualization(iteration, temperature, current_fit, best_fit)
            # if visualize_callback and iteration % 5 == 0:
            #     visualize_callback({
            #         "positions": self.positions.tolist(),
            #         "fitness": self.fitness_history,
            #         "algorithm": "sa",
            #         "env_state": self.env.get_current_state()
            #     })

        return {
            "solution": self.best_solution,
            "metrics": self.env.evaluate_detailed_solution( self.best_solution),
            "agents": {
                "positions": self.positions.tolist(),
                "fitness": self.fitness_history,
                "algorithm": "sa"
            }
        }
    def _generate_neighbor(self):
        """Generate neighbor solution with single-element mutation"""
        neighbor = self.current_solution.copy()
        idx = self.rng.randint(0, self.num_users)
        neighbor[idx] = self.rng.randint(0, self.num_cells)
        return neighbor

    def _update_visualization(self, iteration: int, temp: float, current_fit: float, best_fit: float):
        """Track SA state in 3D space"""
        self.positions = np.vstack([self.positions, [temp, current_fit, best_fit]])
        self.fitness_history.append(best_fit)    
        
