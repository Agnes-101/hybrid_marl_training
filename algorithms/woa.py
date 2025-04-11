# The `WOAOptimization` class implements a particle swarm optimization algorithm for optimizing
# solutions in a custom environment.
import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class WOAOptimization:
    def __init__(self,  env: NetworkEnvironment, kpi_logger=None):
        self.env=env
        self.num_users = env.num_ue
        self.num_cells = env.num_bs
        self.swarm_size = 30
        self.iterations = 20
        self.seed = 42
        self.kpi_logger = kpi_logger
        self.rng = np.random.RandomState(self.seed)
        # Initialize population using the seeded RNG
        self.population = [self.rng.randint(0, self.num_cells, size=self.num_users) for _ in range(self.swarm_size)]
        self.best_solution = max(self.population, key=self.fitness)
        
        # Visualization states
        self.positions = np.empty((0, 3))  # (tabu_usage, diversity, fitness)
        self.fitness_history = []
        
        # Initialize search state
        self.best_solution = max(self.population, key=self.fitness)
        # self.current_solution = self.rng.randint(0, num_cells, num_users)
        # self.best_solution = self.current_solution.copy()
    
    def fitness(self, solution):
        return self.env.evaluate_detailed_solution( solution)["fitness"]
    
    def run(self, visualize_callback: callable = None, kpi_logger=None) -> dict:
        # 🔴 Capture initial state
        original_state = self.env.get_state_snapshot()
        best_fitness = self.fitness(self.best_solution)
        for iteration in range(self.iterations):
            # a = 2 - t * (2 / self.iterations)
            # for i in range(self.swarm_size):
            #     p = self.rng.rand()
            #     new_solution = self.population[i].copy()
            #     if p < 0.5:
            #         if abs(2 * a * self.rng.rand() - a) >= 1:
            #             rand_sol = self.population[self.rng.randint(self.swarm_size)]
            #             for j in range(self.num_users):
            #                 if self.rng.rand() < abs(2 * a * self.rng.rand() - a):
            #                     new_solution[j] = rand_sol[j]
            #         else:
            #             for j in range(self.num_users):
            #                 if self.rng.rand() < 0.5:
            #                     new_solution[j] = self.best_solution[j]
            #     else:
            #         for j in range(self.num_users):
            #             if self.rng.rand() < 0.5:
            #                 new_solution[j] = self.best_solution[j]
            #     if self.fitness(new_solution) > self.fitness(self.population[i]):
            #         self.population[i] = new_solution
            #         if self.fitness(new_solution) > self.fitness(self.best_solution):
            #             self.best_solution = new_solution.copy()
            a = 2 - iteration * (2 / self.iterations)  # Exploration parameter
            
            # Update population
            for i in range(self.swarm_size):
                new_sol = self._update_whale(self.population[i], a)
                if self.fitness(new_sol) > self.fitness(self.population[i]):
                    self.population[i] = new_sol
                    if self.fitness(new_sol) > best_fitness:
                        self.best_solution = new_sol.copy()
                        best_fitness = self.fitness(new_sol)
                        
            # ✅ DE-style logging
            current_metrics = self.env.evaluate_detailed_solution( self.best_solution)
            if self.kpi_logger:
                self.kpi_logger.log_metrics(
                    episode=iteration,
                    phase="metaheuristic",
                    algorithm="woa",
                    metrics=current_metrics
                )
            
            # ✅ Environment update
        # 🔴 Restore environment after optimization
        self.env.set_state_snapshot(original_state)
        self.env.apply_solution(self.best_solution)
        # self.env.step({
        #         f"bs_{bs_id}": np.where(self.best_solution == bs_id)[0].tolist()
        #         for bs_id in range(self.env.num_bs)
        #     })

            # ✅ Visualization updates
            # self._update_visualization(iteration, a)
            # if visualize_callback and iteration % 5 == 0:
            #     visualize_callback({
            #         "positions": self.positions.tolist(),
            #         "fitness": self.fitness_history,
            #         "algorithm": "woa",
            #         "env_state": self.env.get_current_state()
            #     })

        return {
            "solution": self.best_solution,
            "metrics": self.env.evaluate_detailed_solution(self.best_solution),
            "agents": {
                "positions": self.positions.tolist(),
                "fitness": self.fitness_history,
                "algorithm": "woa"
            }
        }

    def _update_whale(self, whale, a):
        """Core WOA update rules"""
        new_sol = whale.copy()
        p = self.rng.rand()
        
        if p < 0.5:
            if abs(2 * a * self.rng.rand() - a) >= 1:
                # Exploration phase
                rand_sol = self.population[self.rng.randint(self.swarm_size)]
                for j in range(self.num_users):
                    if self.rng.rand() < abs(2 * a * self.rng.rand() - a):
                        new_sol[j] = rand_sol[j]
            else:
                # Exploitation phase (bubble-net attacking)
                for j in range(self.num_users):
                    if self.rng.rand() < 0.5:
                        new_sol[j] = self.best_solution[j]
        else:
            # Spiral updating position
            for j in range(self.num_users):
                if self.rng.rand() < 0.5:
                    new_sol[j] = self.best_solution[j]
        return new_sol

    def _update_visualization(self, iteration: int, a: float):
        """Track swarm state in 3D space"""
        # Diversity: mean pairwise hamming distance
        diversity = np.mean([np.sum(a != b) for a in self.population for b in self.population])
        
        # Best fitness
        best_fitness = self.fitness(self.best_solution)
        
        self.positions = np.vstack([self.positions, [a, diversity, best_fitness]])
        self.fitness_history.append(best_fitness)            
       
