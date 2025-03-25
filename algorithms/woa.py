import numpy as np
from envs.custom_channel_env import evaluate_detailed_solution

class WOAOptimization:
    def __init__(self, num_users, num_cells, env, swarm_size=30, iterations=50, seed=None):
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # Initialize population using the seeded RNG
        self.population = [self.rng.randint(0, num_cells, size=num_users) for _ in range(swarm_size)]
        self.best_solution = max(self.population, key=self.fitness)
    
    def fitness(self, solution):
        return evaluate_detailed_solution(self.env, solution)["fitness"]
    
    def optimize(self):
        for t in range(self.iterations):
            a = 2 - t * (2 / self.iterations)
            for i in range(self.swarm_size):
                p = self.rng.rand()
                new_solution = self.population[i].copy()
                if p < 0.5:
                    if abs(2 * a * self.rng.rand() - a) >= 1:
                        rand_sol = self.population[self.rng.randint(self.swarm_size)]
                        for j in range(self.num_users):
                            if self.rng.rand() < abs(2 * a * self.rng.rand() - a):
                                new_solution[j] = rand_sol[j]
                    else:
                        for j in range(self.num_users):
                            if self.rng.rand() < 0.5:
                                new_solution[j] = self.best_solution[j]
                else:
                    for j in range(self.num_users):
                        if self.rng.rand() < 0.5:
                            new_solution[j] = self.best_solution[j]
                if self.fitness(new_solution) > self.fitness(self.population[i]):
                    self.population[i] = new_solution
                    if self.fitness(new_solution) > self.fitness(self.best_solution):
                        self.best_solution = new_solution.copy()
        return self.best_solution
