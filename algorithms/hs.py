import numpy as np
from envs.custom_channel_env import evaluate_detailed_solution

class HarmonySearchOptimization:
    def __init__(self, num_users, num_cells, env, memory_size=30, iterations=50, HMCR=0.9, PAR=0.3, seed=None):
        self.num_users = num_users
        self.num_cells = num_cells
        self.env = env
        self.memory_size = memory_size
        self.iterations = iterations
        self.HMCR = HMCR
        self.PAR = PAR
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # Initialize harmony memory using the seeded RNG
        self.harmony_memory = [self.rng.randint(0, num_cells, size=num_users) for _ in range(memory_size)]
    
    def fitness(self, solution):
        return evaluate_detailed_solution(self.env, solution)["fitness"]
    
    def optimize(self):
        for _ in range(self.iterations):
            new_harmony = np.zeros(self.num_users, dtype=int)
            for j in range(self.num_users):
                if self.rng.rand() < self.HMCR:
                    # Use seeded RNG to select a value from harmony memory
                    values = [h[j] for h in self.harmony_memory]
                    new_harmony[j] = self.rng.choice(values)
                    if self.rng.rand() < self.PAR:
                        new_harmony[j] = self.rng.randint(0, self.num_cells)
                else:
                    new_harmony[j] = self.rng.randint(0, self.num_cells)
            # Identify the worst harmony in the memory
            fitness_values = [self.fitness(h) for h in self.harmony_memory]
            worst_index = np.argmin(fitness_values)
            if self.fitness(new_harmony) > self.fitness(self.harmony_memory[worst_index]):
                self.harmony_memory[worst_index] = new_harmony
        return max(self.harmony_memory, key=self.fitness)
