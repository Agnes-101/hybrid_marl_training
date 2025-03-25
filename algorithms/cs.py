import numpy as np
import random
from envs.custom_channel_env import evaluate_detailed_solution

class CSOptimization:
    def __init__(self, num_users, num_cells, env, colony_size=30, iterations=50, pa=0.25, seed=None):
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.colony_size = colony_size
        self.iterations = iterations
        self.pa = pa
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # Initialize population using the seeded RNG
        self.population = [self.rng.randint(0, num_cells, size=num_users) for _ in range(colony_size)]
    
    def fitness(self, solution):
        return evaluate_detailed_solution(self.env, solution)["fitness"]
    
    def levy_flight(self):
        # Generate a levy flight step using the seeded RNG
        return self.rng.randint(-1, 2, size=self.env.num_users)
    
    def optimize(self):
        for _ in range(self.iterations):
            for i in range(self.colony_size):
                new_solution = self.population[i].copy()
                step = self.levy_flight()
                new_solution = (new_solution + step) % self.env.num_cells
                if self.fitness(new_solution) > self.fitness(self.population[i]):
                    self.population[i] = new_solution
            fitnesses = [self.fitness(sol) for sol in self.population]
            indices = np.argsort(fitnesses)
            num_abandon = int(self.pa * self.colony_size)
            for idx in indices[:num_abandon]:
                self.population[idx] = self.rng.randint(0, self.env.num_cells, size=self.env.num_users)
        return max(self.population, key=self.fitness)
