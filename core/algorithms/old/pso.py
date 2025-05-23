import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class PSOOptimization:
    def __init__(self, env: NetworkEnvironment, kpi_logger=None):
        self.env=env
        self.num_users = env.num_ue
        self.num_cells = env.num_bs
        self.swarm_size = 30
        self.iterations = 20
        self.c1 = 1
        self.c2 = 1
        self.w = 0.5
        self.seed = 42
        self.rng = np.random.RandomState(self.seed)
        # Initialize positions using the seeded RNG
        self.population = [self.rng.randint(0, self.num_cells, size=self.num_users) for _ in range(self.swarm_size)]
        self.pbest = self.population.copy()
        self.gbest = max(self.population, key=self.fitness)
        self.kpi_logger = kpi_logger
        
        # Visualization states
        self.positions = np.empty((0, 3))  # (convergence, diversity, fitness)
        self.fitness_history = []
    
    def fitness(self, solution):
        return self.env.evaluate_detailed_solution(solution)["fitness"]
    
    def run(self, visualize_callback: callable = None, kpi_logger=None) -> dict:
        # 🔴 Capture initial state
        original_state = self.env.get_state_snapshot()
        best_fitness = self.fitness(self.gbest)
        for iteration in range(self.iterations):
            # for i in range(self.swarm_size):
            #     new_solution = self.population[i].copy()
            #     for j in range(self.num_users):
            #         if self.rng.rand() < self.c1 * 0.5:
            #             new_solution[j] = self.pbest[i][j]
            #         if self.rng.rand() < self.c2 * 0.5:
            #             new_solution[j] = self.gbest[j]
            #         if self.rng.rand() < self.w * 0.1:
            #             new_solution[j] = self.rng.randint(0, self.num_cells)
            #     if self.fitness(new_solution) > self.fitness(self.population[i]):
            #         self.population[i] = new_solution
            #         self.pbest[i] = new_solution
            #         if self.fitness(new_solution) > self.fitness(self.gbest):
            #             self.gbest = new_solution
            # Update particles
            new_population = []
            new_pbest = []
            for i in range(self.swarm_size):
                new_sol = self._update_particle(self.population[i], self.pbest[i])
                new_fitness = self.fitness(new_sol)
                
                # Update personal best
                if new_fitness > self.fitness(self.pbest[i]):
                    new_pbest.append(new_sol)
                    # Update global best
                    if new_fitness > best_fitness:
                        best_fitness = new_fitness
                        self.gbest = new_sol.copy()
                else:
                    new_pbest.append(self.pbest[i])
                
                new_population.append(new_sol)
            
            self.population = new_population
            self.pbest = new_pbest
            # ✅ DE-style logging
            current_metrics = self.env.evaluate_detailed_solution( self.gbest)
            if self.kpi_logger:
                self.kpi_logger.log_metrics(
                    episode=iteration,
                    phase="metaheuristic",
                    algorithm="pso",
                    metrics=current_metrics
                )
                
            # ✅ Visualization updates
            self._update_visualization(iteration)
            if visualize_callback and iteration % 5 == 0:
                visualize_callback({
                    "positions": self.visual_positions.tolist(),
                    "fitness": self.fitness_history,
                    "algorithm": "pso",
                    "env_state": self.env.get_current_state()
                })
            
            # ✅ Environment update
        # 🔴 Restore environment after optimization
        self.env.set_state_snapshot(original_state)
        self.env.apply_solution(self.gbest)
        # self.env.step({
        #         f"bs_{bs_id}": np.where(self.gbest == bs_id)[0].tolist()
        #         for bs_id in range(self.env.num_bs)
        #     })

           
        return {
            "solution": self.gbest,
            "metrics": self.env.evaluate_detailed_solution(self.gbest),
            "agents": {
                "positions": self.positions.tolist(),
                "fitness": self.fitness_history,
                "algorithm": "pso"
            }
        }
        
    def _update_particle(self, position, pbest):
        """Core PSO update rules with discrete adaptation"""
        new_sol = position.copy()
        for j in range(self.num_users):
            # Cognitive component
            if self.rng.rand() < self.c1 * 0.5:
                new_sol[j] = pbest[j]
            # Social component
            if self.rng.rand() < self.c2 * 0.5:
                new_sol[j] = self.gbest[j]
            # Random perturbation
            if self.rng.rand() < self.w * 0.1:
                new_sol[j] = self.rng.randint(0, self.num_cells)
        return new_sol

    def _update_visualization(self, iteration: int):
        """Track swarm convergence and diversity"""
        # Convergence: percentage of particles matching gbest
        convergence = np.mean([np.all(sol == self.gbest) for sol in self.positions])
        
        # Diversity: mean pairwise hamming distance
        diversity = np.mean([np.sum(a != b) for a in self.positions for b in self.positions])
        
        # Best fitness
        best_fitness = self.fitness(self.gbest)
        
        self.positions = np.vstack([
            self.positions,
            [convergence, diversity, best_fitness]
        ])
        self.fitness_history.append(best_fitness)    
