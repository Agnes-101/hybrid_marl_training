import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class FireflyOptimization:
    def __init__(self, env, kpi_logger=None ):
        self.env = env
        self.num_users = env.num_ue
        self.num_cells = env.num_bs
        
        self.population_size = 30
        self.iterations = 20
        self.beta0 = 1
        self.gamma = 1
        self.seed = 42
        self.rng = np.random.RandomState(self.seed)
        self.kpi_logger = kpi_logger
        # Initialize population using the seeded RNG
        self.population = [self.rng.randint(0, self.num_cells, size=self.num_users) for _ in range(self.population_size)]
        # Visualization states
        self.positions = np.empty((0, 3))  # (intensity, diversity, fitness)
        self.fitness_history = []
        self.best_solution = None
        
    def fitness(self, solution):# , env: NetworkEnvironment
        return self.env.evaluate_detailed_solution( solution)["fitness"]
    
    def distance(self, sol1, sol2):
        return np.sum(sol1 != sol2)  # Hamming distance
    
    def run(self, env: NetworkEnvironment, visualize_callback: callable = None, kpi_logger=None) -> dict:
        best_fitness = -np.inf
        for iteration in range(self.iterations):
            # Track iteration-best metrics
            current_best_sol = max(self.population, key=self.fitness)
            current_best_metrics = env.evaluate_detailed_solution(current_best_sol)
            # ✅ DE-style logging
            if self.kpi_logger:
                self.kpi_logger.log_metrics(
                    episode=iteration,
                    phase="metaheuristic",
                    algorithm="firefly",
                    metrics=current_best_metrics
                )
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if self.fitness(self.population[j]) > self.fitness(self.population[i]):
                        r = self.distance(self.population[i], self.population[j])
                        beta = self.beta0 * np.exp(-self.gamma * r**2)
                        # new_solution = self.population[i].copy()
                        new_solution = self._attraction_move(self.population[i], self.population[j], beta)
                        # for k in range(self.num_users):
                        #     # Use the seeded RNG for generating random numbers
                        #     if self.rng.rand() < beta:
                        #         new_solution[k] = self.population[j][k]
                                
                        if self.fitness(new_solution) > self.fitness(self.population[i]):
                            self.population[i] = new_solution
                            
                # Environment update
            self.best_solution = max(self.population, key=self.fitness)
            env.apply_solution(self.best_solution)
            env.step({
                f"bs_{bs_id}": np.where(self.best_solution == bs_id)[0].tolist()
                for bs_id in range(env.num_bs)
            })
            
            # ✅ Visualization updates
            # self._update_visualization(iteration)
            # if visualize_callback and iteration % 5 == 0:
            #     visualize_callback({
            #         "positions": self.positions.tolist(),
            #         "fitness": self.fitness_history,
            #         "algorithm": "firefly",
            #         "env_state": env.get_current_state()
            #     })
        return {
            "solution": self.best_solution,
            "metrics": current_best_metrics,# env.evaluate_detailed_solution( self.best_solution),
            "agents": {
                "positions": self.positions.tolist(),
                "fitness": self.fitness_history,
                "algorithm": "firefly"
            }
        }
    def _attraction_move(self, sol_i, sol_j, beta):
        """Vectorized attraction movement"""
        mask = self.rng.rand(self.num_users) < beta
        new_sol = sol_i.copy()
        new_sol[mask] = sol_j[mask]
        return new_sol

    def _update_visualization(self, iteration: int):
        """DE-compatible 3D visualization state"""
        # Intensity: average brightness of population
        intensities = [self.fitness(sol) for sol in self.population]
        avg_intensity = np.mean(intensities)
        
        # Diversity: mean pairwise distance
        diversity = np.mean([self.distance(a,b) for a in self.population for b in self.population])
        
        # Fitness: best solution's fitness
        best_fitness = self.fitness(self.best_solution)
        
        self.positions = np.vstack([self.positions, [avg_intensity, diversity, best_fitness]])
        self.fitness_history.append(best_fitness)

    def fitness(self, solution):
        return self.env.evaluate_detailed_solution(solution)["fitness"]

    def distance(self, sol1, sol2):
        return np.sum(sol1 != sol2)  # Hamming distance
