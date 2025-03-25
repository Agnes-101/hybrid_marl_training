import numpy as np
import random
from envs.custom_channel_env import evaluate_detailed_solution

class BatOptimization:
    def __init__(self, num_users, num_cells, env, population_size=30, iterations=50, 
                freq_min=0, freq_max=2, alpha=0.9, gamma=0.9, seed=None):
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.population_size = population_size
        self.iterations = iterations
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.alpha = alpha
        self.gamma = gamma
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # Initialize population using the seeded RNG
        self.population = [self.rng.randint(0, num_cells, size=num_users) for _ in range(population_size)]
        self.loudness = [1.0 for _ in range(population_size)]
        self.pulse_rate = [0.5 for _ in range(population_size)]
        self.best = max(self.population, key=self.fitness)
    
    def fitness(self, solution):
        return evaluate_detailed_solution(self.env, solution)["fitness"]
    
    def optimize(self):
        for t in range(self.iterations):
            for i in range(self.population_size):
                # Use the seeded RNG to generate frequency
                freq = self.freq_min + (self.freq_max - self.freq_min) * self.rng.rand()
                new_solution = self.population[i].copy()
                for j in range(self.num_users):
                    # Use seeded RNG for decision making
                    if self.rng.rand() > self.pulse_rate[i]:
                        new_solution[j] = self.rng.randint(0, self.num_cells)
                if self.fitness(new_solution) > self.fitness(self.population[i]) and self.rng.rand() < self.loudness[i]:
                    self.population[i] = new_solution
                    self.loudness[i] *= self.alpha
                    self.pulse_rate[i] = self.pulse_rate[i] * (1 - np.exp(-self.gamma * t))
                if self.fitness(self.population[i]) > self.fitness(self.best):
                    self.best = self.population[i]
        return self.best
