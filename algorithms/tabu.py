import numpy as np
from envs.custom_channel_env import evaluate_detailed_solution

class TabuSearchOptimization:
    def __init__(self, num_users, num_cells, env, iterations=50, tabu_size=10, seed=None):
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.iterations = iterations
        self.tabu_size = tabu_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # Initialize current solution using the seeded RNG
        self.current_solution = self.rng.randint(0, num_cells, size=num_users)
        self.tabu_list = []
    
    def fitness(self, solution):
        return evaluate_detailed_solution(self.env, solution)["fitness"]
    
    def optimize(self):
        best_solution = self.current_solution.copy()
        best_fitness = self.fitness(best_solution)
        for _ in range(self.iterations):
            neighbors = []
            for i in range(self.num_users):
                for new_bs in range(self.num_cells):
                    if new_bs != self.current_solution[i]:
                        neighbor = self.current_solution.copy()
                        neighbor[i] = new_bs
                        if tuple(neighbor) not in self.tabu_list:
                            neighbors.append(neighbor)
            if not neighbors:
                break
            neighbor_fitness = [self.fitness(n) for n in neighbors]
            best_neighbor = neighbors[np.argmax(neighbor_fitness)]
            current_fitness = self.fitness(best_neighbor)
            if current_fitness > best_fitness:
                best_solution = best_neighbor.copy()
                best_fitness = current_fitness
            self.current_solution = best_neighbor.copy()
            self.tabu_list.append(tuple(self.current_solution))
            if len(self.tabu_list) > self.tabu_size:
                self.tabu_list.pop(0)
        return best_solution
