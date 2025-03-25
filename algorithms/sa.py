import numpy as np
from envs.custom_channel_env import evaluate_detailed_solution

class SAOptimization:
    def __init__(self, num_users, num_cells, env, iterations=100, initial_temp=100, cooling_rate=0.95, seed=None):
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.iterations = iterations
        self.temperature = initial_temp
        self.cooling_rate = cooling_rate
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # Initialize current solution using the seeded RNG
        self.current_solution = self.rng.randint(0, num_cells, size=num_users)
        
    def fitness(self, solution):
        return evaluate_detailed_solution(self.env, solution)["fitness"]
    
    def optimize(self):
        current_fit = self.fitness(self.current_solution)
        best_solution = self.current_solution.copy()
        best_fit = current_fit
        for _ in range(self.iterations):
            neighbor = self.current_solution.copy()
            idx = self.rng.randint(0, self.num_users)
            neighbor[idx] = self.rng.randint(0, self.num_cells)
            neighbor_fit = self.fitness(neighbor)
            delta = neighbor_fit - current_fit
            if delta > 0 or self.rng.rand() < np.exp(delta / self.temperature):
                self.current_solution = neighbor
                current_fit = neighbor_fit
                if neighbor_fit > best_fit:
                    best_solution = neighbor.copy()
                    best_fit = neighbor_fit
            self.temperature *= self.cooling_rate
        return best_solution
