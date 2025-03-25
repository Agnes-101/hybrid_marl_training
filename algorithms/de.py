import numpy as np
from envs.custom_channel_env import evaluate_detailed_solution

class DEOptimization:
    def __init__(self, num_users, num_cells, env, population_size=30, iterations=50, F=0.8, CR=0.9, seed=None):
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.population_size = population_size
        self.iterations = iterations
        self.F = F
        self.CR = CR
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # Initialize population using the seeded RNG
        self.population = [self.rng.randint(0, num_cells, size=num_users) for _ in range(population_size)]
    
    def fitness(self, solution):
        return evaluate_detailed_solution(self.env, solution)["fitness"]
    
    def optimize(self):
        for _ in range(self.iterations):
            new_population = []
            for i in range(self.population_size):
                # Create a list of indices excluding the current index i
                indices = [idx for idx in range(self.population_size) if idx != i]
                # Randomly select three distinct indices using the seeded RNG
                selected = self.rng.choice(indices, size=3, replace=False)
                a, b, c = selected
                # Create donor vector using DE mutation formula
                donor = self.population[a] + self.F * (self.population[b] - self.population[c])
                donor = np.clip(np.round(donor).astype(int), 0, self.num_cells - 1)
                trial = self.population[i].copy()
                # Crossover: use self.rng for generating random numbers
                for j in range(self.num_users):
                    if self.rng.rand() < self.CR:
                        trial[j] = donor[j]
                # Selection: if trial has better fitness, replace individual
                if self.fitness(trial) > self.fitness(self.population[i]):
                    new_population.append(trial)
                else:
                    new_population.append(self.population[i])
            self.population = new_population
        return max(self.population, key=self.fitness)
