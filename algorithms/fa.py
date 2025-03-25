import numpy as np
from envs.custom_channel_env import evaluate_detailed_solution

class FireflyOptimization:
    def __init__(self, num_users, num_cells, env, population_size=30, iterations=50, beta0=1, gamma=1, seed=None):
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.population_size = population_size
        self.iterations = iterations
        self.beta0 = beta0
        self.gamma = gamma
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # Initialize population using the seeded RNG
        self.population = [self.rng.randint(0, num_cells, size=num_users) for _ in range(population_size)]
    
    def fitness(self, solution):
        return evaluate_detailed_solution(self.env, solution)["fitness"]
    
    def distance(self, sol1, sol2):
        return np.sum(sol1 != sol2)  # Hamming distance
    
    def optimize(self):
        for _ in range(self.iterations):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if self.fitness(self.population[j]) > self.fitness(self.population[i]):
                        r = self.distance(self.population[i], self.population[j])
                        beta = self.beta0 * np.exp(-self.gamma * r**2)
                        new_solution = self.population[i].copy()
                        for k in range(self.num_users):
                            # Use the seeded RNG for generating random numbers
                            if self.rng.rand() < beta:
                                new_solution[k] = self.population[j][k]
                        if self.fitness(new_solution) > self.fitness(self.population[i]):
                            self.population[i] = new_solution
        return max(self.population, key=self.fitness)
