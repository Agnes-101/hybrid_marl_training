# algorithms/de.py
import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class DEOptimization:
    def __init__(self):
        """Differential Evolution for 6G UE-BS Association"""
        # Optimization parameters
        self.population_size = 30
        self.iterations = 50
        self.F = 0.8  # Mutation factor
        self.CR = 0.9  # Crossover rate
        
        # State tracking (standardized attributes)
        self.positions = np.empty((0, 3))  # [x, y, fitness]
        self.fitness = np.array([])
        self.velocity = None  # For compatibility with PSO-based visualizations
        
        # Internal state
        self.population = None
        self.best_solution = None
        self._rng = np.random.RandomState()

    def run(self, env: NetworkEnvironment, visualize_callback: callable = None) -> dict:
        """Execute DE optimization with universal state tracking"""
        self._initialize_population(env)
        self.best_solution = self.population[0]
        self.fitness = np.zeros(self.iterations)
        
        for iteration in range(self.iterations):
            self._adapt_parameters(iteration)
            new_population = []
            
            for i in range(self.population_size):
                trial = self._create_trial_vector(i, env)
                trial = self._repair_solution(trial, env)
                
                current_fit = env.evaluate_detailed_solution(self.population[i])["fitness"]
                trial_fit = env.evaluate_detailed_solution(trial)["fitness"]
                
                if trial_fit > current_fit:
                    new_population.append(trial)
                    if trial_fit > self.fitness[iteration]:
                        self.best_solution = trial.copy()
                        self.fitness[iteration] = trial_fit
                else:
                    new_population.append(self.population[i])
            
            self.population = new_population
            self._update_visual_state(env)
            
            if visualize_callback and iteration % 5 == 0:
                visualize_callback()

        return {
            "solution": self.best_solution,
            "metrics": env.evaluate_detailed_solution(self.best_solution),
            "agents": {"positions": self.positions, "fitness": self.fitness}
        }

    def _initialize_population(self, env: NetworkEnvironment):
        """Generate random UE-BS associations"""
        self.population = [
            self._rng.randint(0, env.num_bs, env.num_ue)
            for _ in range(self.population_size)
        ]

    def _create_trial_vector(self, target_idx: int, env: NetworkEnvironment) -> np.ndarray:
        """DE mutation and crossover operations"""
        candidates = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = self._rng.choice(candidates, 3, False)
        
        mutant = np.clip(
            self.population[a] + self.F * (self.population[b] - self.population[c]),
            0, env.num_bs - 1
        ).astype(int)
        
        trial = self.population[target_idx].copy()
        cross_mask = self._rng.rand(env.num_ue) < self.CR
        trial[cross_mask] = mutant[cross_mask]
        
        return trial

    def _repair_solution(self, trial: np.ndarray, env: NetworkEnvironment) -> np.ndarray:
        """Ensure solutions respect BS capacity constraints"""
        bs_counts = np.bincount(trial, minlength=env.num_bs)
        
        # Get list of all BS capacities
        capacities = np.array([bs.capacity for bs in env.base_stations])
        
        # Identify overloaded BS indices
        overloaded_bs_ids = np.where(bs_counts > capacities)[0]
        
        for bs_id in overloaded_bs_ids:
            excess = bs_counts[bs_id] - capacities[bs_id]
            ue_indices = np.where(trial == bs_id)[0]
            np.random.shuffle(ue_indices)
            
            for idx in ue_indices[:excess]:
                available_bs = [b.id for b in env.base_stations 
                            if bs_counts[b.id] < b.capacity]
                if available_bs:
                    trial[idx] = np.random.choice(available_bs)
                    bs_counts[bs_id] -= 1
                    bs_counts[trial[idx]] += 1
        return trial

    def _adapt_parameters(self, iteration: int):
        """Self-adaptive parameter adjustment"""
        progress = iteration / self.iterations
        self.F = 0.8 - 0.3 * progress  # Linear decay
        self.CR = 0.9 - 0.2 * (1 - progress)  # Curved adaptation

    def _update_visual_state(self, env: NetworkEnvironment):
        """Prepare universal visualization state"""
        # Convert best solution to spatial coordinates
        bs_positions = {bs.id: bs.position.numpy() for bs in env.base_stations}
        current_fitness = env.evaluate_detailed_solution(self.best_solution)["fitness"]
        
        self.positions = np.array([
            [bs_positions[bs_id][0], bs_positions[bs_id][1], current_fitness]
            for bs_id in self.best_solution
        ])
        
        # Maintain fitness history (truncate/pad to match population size)
        self.fitness = np.concatenate((
            self.fitness,
            np.full(self.population_size - len(self.fitness), current_fitness)
        ))[:self.population_size]



# import numpy as np
# from envs.custom_channel_env import evaluate_detailed_solution

# class DEOptimization:
#     def __init__(self, num_users, num_cells, env, population_size=30, iterations=50, F=0.8, CR=0.9, seed=None):
#         self.num_users = num_users
#         self.num_cells = num_cells
#         self.env = env
#         self.population_size = population_size
#         self.iterations = iterations
#         self.F = F
#         self.CR = CR
#         self.seed = seed
#         self.rng = np.random.RandomState(seed)
#         # Initialize population using the seeded RNG
#         self.population = [self.rng.randint(0, num_cells, size=num_users) for _ in range(population_size)]
    
#     def fitness(self, solution):
#         return evaluate_detailed_solution(self.env, solution)["fitness"]
    
#     def optimize(self):
#         for _ in range(self.iterations):
#             new_population = []
#             for i in range(self.population_size):
#                 # Create a list of indices excluding the current index i
#                 indices = [idx for idx in range(self.population_size) if idx != i]
#                 # Randomly select three distinct indices using the seeded RNG
#                 selected = self.rng.choice(indices, size=3, replace=False)
#                 a, b, c = selected
#                 # Create donor vector using DE mutation formula
#                 donor = self.population[a] + self.F * (self.population[b] - self.population[c])
#                 donor = np.clip(np.round(donor).astype(int), 0, self.num_cells - 1)
#                 trial = self.population[i].copy()
#                 # Crossover: use self.rng for generating random numbers
#                 for j in range(self.num_users):
#                     if self.rng.rand() < self.CR:
#                         trial[j] = donor[j]
#                 # Selection: if trial has better fitness, replace individual
#                 if self.fitness(trial) > self.fitness(self.population[i]):
#                     new_population.append(trial)
#                 else:
#                     new_population.append(self.population[i])
#             self.population = new_population
#         return max(self.population, key=self.fitness)
