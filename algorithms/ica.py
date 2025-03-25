import numpy as np
from envs.custom_channel_env import evaluate_detailed_solution

class ICAOptimization:
    def __init__(self, num_users, num_cells, env, population_size=30, imperialist_count=5, iterations=50, seed=None):
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.population_size = population_size
        self.imperialist_count = imperialist_count
        self.iterations = iterations
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # Initialize population using the seeded RNG
        self.population = [self.rng.randint(0, num_cells, size=num_users) for _ in range(population_size)]
    
    def fitness(self, solution):
        return evaluate_detailed_solution(self.env, solution)["fitness"]
    
    def optimize(self):
        sorted_population = sorted(self.population, key=self.fitness, reverse=True)
        imperialists = sorted_population[:self.imperialist_count]
        colonies = sorted_population[self.imperialist_count:]
        for _ in range(self.iterations):
            for i in range(len(colonies)):
                imp = imperialists[i % self.imperialist_count]
                colony = colonies[i].copy()
                # Use the seeded RNG for selecting a random index
                idx = self.rng.randint(0, self.num_users)
                colony[idx] = imp[idx]
                if self.fitness(colony) > self.fitness(colonies[i]):
                    colonies[i] = colony
            self.population = imperialists + colonies
            self.population = sorted(self.population, key=self.fitness, reverse=True)
            imperialists = self.population[:self.imperialist_count]
            colonies = self.population[self.imperialist_count:]
        return imperialists[0]
