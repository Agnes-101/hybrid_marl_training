import numpy as np
from envs.custom_channel_env import evaluate_detailed_solution

class GAOptimization:
    def __init__(self, num_users, num_cells, env, population_size=30, generations=50, mutation_rate=0.1, seed=None):
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # Initialize population using the seeded RNG
        self.population = [self.rng.randint(0, num_cells, size=num_users) for _ in range(population_size)]
    
    def fitness(self, solution):
        return evaluate_detailed_solution(self.env, solution)["fitness"]
    
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
    
    def optimize(self):
        for _ in range(self.generations):
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1))
                if len(new_population) < self.population_size:
                    new_population.append(self.mutate(child2))
            self.population = new_population
        return max(self.population, key=self.fitness)
