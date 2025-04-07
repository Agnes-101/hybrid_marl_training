import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class GAOptimization:
    def __init__(self, env, kpi_logger=None):
        self.env=env
        self.num_users = env.num_ue
        self.num_cells = env.num_bs
        self.population_size = 30
        self.generations = 50
        self.mutation_rate = 0.1
        self.seed = 42
        self.rng = np.random.RandomState(self.seed)
        # Visualization states
        self.positions = np.empty((0, 3))  # (diversity, avg_fitness, best_fitness)
        self.fitness_history = []
        self.kpi_logger = kpi_logger
        # Initialize population using the seeded RNG
        self.population = [self.rng.randint(0, self.num_cells, size=self.num_users) for _ in range(self.population_size)]
    
    def fitness(self, solution):
        return self.env.evaluate_detailed_solution(solution)["fitness"]
    
    def tournament_selection(self, k=3):
        # Select k random indices from the population without replacement
        indices = self.rng.choice(len(self.population), size=k, replace=False)
        participants = [self.population[i] for i in indices]
        return max(participants, key=self.fitness)
    
    def crossover(self, parent1, parent2):
        # Choose a random crossover point using the seeded RNG
        point = self.rng.randint(1, self.num_users)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    
    def mutate(self, solution):
        for i in range(self.num_users):
            if self.rng.rand() < self.mutation_rate:
                solution[i] = self.rng.randint(0, self.num_cells)
        return solution
    
    def run(self, env: NetworkEnvironment, visualize_callback: callable = None, kpi_logger=None) -> dict:
        best_solution = None
        best_fitness = -np.inf
        
        for generation in range(self.generations):
            # Evaluate population
            fitness_values = [self.fitness(sol) for sol in self.population]
            current_best_idx = np.argmax(fitness_values)
            current_best_sol = self.population[current_best_idx]
            current_metrics = self.env.evaluate_detailed_solution(current_best_sol)
            
            # ✅ DE-style logging
            if self.kpi_logger:
                self.kpi_logger.log_metrics(
                    episode=generation,
                    phase="metaheuristic",
                    algorithm="ga",
                    metrics=current_metrics
                )
                
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                child1, child2 = self.crossover(parent1, parent2)
                # new_population.append(self.mutate(child1))
                # if len(new_population) < self.population_size:
                #     new_population.append(self.mutate(child2))
                new_population.extend([self.mutate(child1), self.mutate(child2)])
            # self.population = new_population
            self.population = new_population[:self.population_size]  # Trim excess
            # ✅ Environment update
            self.env.apply_solution(best_solution)
            self.env.step({
                f"bs_{bs_id}": np.where(best_solution == bs_id)[0].tolist()
                for bs_id in range(self.env.num_bs)
            })

            # ✅ Visualization updates
            # self._update_visualization(generation)
            # if visualize_callback and generation % 5 == 0:
            #     visualize_callback({
            #         "positions": self.positions.tolist(),
            #         "fitness": self.fitness_history,
            #         "algorithm": "ga",
            #         "env_state": self.env.get_current_state()
            #     })
        return {
            "solution": best_solution,
            "metrics": self.env.evaluate_detailed_solution(best_solution),
            "agents": {
                "positions": self.positions.tolist(),
                "fitness": self.fitness_history,
                "algorithm": "ga"
            }
        }
        
    def _update_visualization(self, generation: int):
        """Track population diversity and fitness landscape"""
        # Population diversity: mean pairwise hamming distance
        diversity = np.mean([self._hamming_distance(a,b) 
                        for a in self.population for b in self.population])
        
        # Fitness metrics
        fitness_values = [self.fitness(sol) for sol in self.population]
        avg_fitness = np.mean(fitness_values)
        best_fitness = np.max(fitness_values)
        
        self.positions = np.vstack([self.positions, [diversity, avg_fitness, best_fitness]])
        self.fitness_history.append(best_fitness)

    def _hamming_distance(self, sol1, sol2):
        return np.sum(sol1 != sol2)    
