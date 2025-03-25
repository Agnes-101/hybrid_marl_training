# algorithms/gwo.py
import numpy as np
from envs.custom_channel_env import evaluate_detailed_solution

class GWOOptimization:
    def __init__(self, num_users, num_cells, env, swarm_size=30, iterations=50,
                 a_initial=2.0, a_decay=0.04, seed=None):
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.a_initial = a_initial
        self.a_decay = a_decay
        self.rng = np.random.RandomState(seed)
        
        # Initialize population using RNG
        self.population = [self.rng.randint(0, num_cells, size=num_users)
                          for _ in range(swarm_size)]
        self.alpha = None
        self.beta = None
        self.delta = None
        self.update_leaders()

    def fitness(self, solution):
        return evaluate_detailed_solution(self.env, solution)["fitness"]

    def update_leaders(self):
        sorted_pop = sorted(self.population, key=lambda s: self.fitness(s), reverse=True)
        self.alpha = sorted_pop[0].copy()
        self.beta = sorted_pop[1].copy() if len(sorted_pop) > 1 else sorted_pop[0].copy()
        self.delta = sorted_pop[2].copy() if len(sorted_pop) > 2 else sorted_pop[0].copy()

    def optimize(self):
        for t in range(self.iterations):
            a = self.a_initial - t * self.a_decay  # Use configured parameters
            new_population = []
            for i in range(self.swarm_size):
                new_solution = self.population[i].copy()
                for j in range(self.num_users):
                    # Use RNG instead of np.random
                    r1, r2 = self.rng.rand(), self.rng.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha[j] - self.population[i][j])
                    X1 = self.alpha[j] - A1 * D_alpha
                    
                    r1, r2 = self.rng.rand(), self.rng.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta[j] - self.population[i][j])
                    X2 = self.beta[j] - A2 * D_beta
                    
                    r1, r2 = self.rng.rand(), self.rng.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta[j] - self.population[i][j])
                    X3 = self.delta[j] - A3 * D_delta
                    
                    new_val = int(round((X1 + X2 + X3) / 3))
                    new_solution[j] = np.clip(new_val, 0, self.env.num_cells - 1)
                new_population.append(new_solution)
            self.population = new_population
            self.update_leaders()
        return self.alpha