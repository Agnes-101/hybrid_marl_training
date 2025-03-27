# algorithms/bat.py
import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class BatOptimization:
    def __init__(self, seed=None):
        """Hardcoded parameters for hybrid training system with decoupled initialization."""
        # Optimization parameters
        self.population_size = 30
        self.iterations = 50
        self.freq_min = 0
        self.freq_max = 2
        self.alpha = 0.9
        self.gamma = 0.9
        self.seed = seed if seed is not None else np.random.randint(0, 10000)
        
        # Visualization states
        self.positions = np.empty((0, 2))  # Bat positions in 2D space
        self.fitness_history = []
        
        # Initialize the random generator
        self.rng = np.random.RandomState(self.seed)
        
        # Environment will be set in run()
        self.env = None

    def run(self, env: NetworkEnvironment) -> dict:
        """Main interface for hybrid training system."""
        self.env = env  # Store the environment for use in visualization updates
        num_ue = env.num_ue
        num_bs = env.num_bs
        
        # Initialize population and tracking
        self.population = [
            self.rng.randint(0, num_bs, size=num_ue)
            for _ in range(self.population_size)
        ]
        self.loudness = np.full(self.population_size, 1.0)
        self.pulse_rate = np.full(self.population_size, 0.5)
        self.best_solution = None
        best_fitness = -np.inf

        for iteration in range(self.iterations):
            # Generate new solutions for each bat in the population
            for i in range(self.population_size):
                freq = self.freq_min + (self.freq_max - self.freq_min) * self.rng.rand()
                new_sol = self._modify_solution(self.population[i], freq)
                new_fitness = env.evaluate_detailed_solution(new_sol)["fitness"]
                
                # Evaluate current fitness
                current_fitness = env.evaluate_detailed_solution(self.population[i])["fitness"]
                # Update if the new solution is better and passes the pulse rate check
                if new_fitness > current_fitness and self.rng.rand() < self.loudness[i]:
                    self.population[i] = new_sol
                    self.loudness[i] *= self.alpha
                    self.pulse_rate[i] *= (1 - np.exp(-self.gamma * iteration))
                
                # Track best solution across the population
                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    self.best_solution = new_sol.copy()
                    
            # Update visualization states after each iteration
            self._update_visualization(iteration)

        return {
            "solution": self.best_solution,
            "metrics": env.evaluate_detailed_solution(self.best_solution),
            "agents": {
                "positions": self.positions,
                "fitness": self.fitness_history
            }
        }

    def _modify_solution(self, solution: np.ndarray, freq: float) -> np.ndarray:
        """Generate new solution using bat algorithm rules."""
        new_sol = solution.copy()
        # Here we use the average pulse rate as a global decision factor
        global_pulse = self.pulse_rate.mean()
        for user in range(len(solution)):
            if self.rng.rand() > global_pulse:  # Global pulse rate average decision
                # Replace with a random base station index
                new_sol[user] = self.rng.randint(0, len(np.unique(solution)))
        return new_sol

    def _update_visualization(self, iteration: int):
        """Track positions and fitness for visualization."""
        # Convert best solution into a 2D coordinate using mean and std as a simplified PCA
        x = np.mean(self.best_solution)
        y = np.std(self.best_solution)
        
        # Append current position data
        self.positions = np.vstack([self.positions, [x, y]])
        # Record the best fitness from the current iteration using the stored environment
        current_fitness = self.env.evaluate_detailed_solution(self.best_solution)["fitness"]
        self.fitness_history.append(current_fitness)
        # Optionally, print iteration details for debugging
        print(f"Iteration {iteration}: Best Fitness = {current_fitness:.4f}, Position = ({x:.2f}, {y:.2f})")


# import numpy as np
# import random
# from envs.custom_channel_env import evaluate_detailed_solution

# class BatOptimization:
#     def __init__(self, num_users, num_cells, env, population_size=30, iterations=50, 
#                 freq_min=0, freq_max=2, alpha=0.9, gamma=0.9, seed=None):
#         self.num_users = num_users
#         self.num_cells = num_cells
#         self.env = env
#         self.population_size = population_size
#         self.iterations = iterations
#         self.freq_min = freq_min
#         self.freq_max = freq_max
#         self.alpha = alpha
#         self.gamma = gamma
#         self.seed = seed
#         self.rng = np.random.RandomState(seed)
#         # Initialize population using the seeded RNG
#         self.population = [self.rng.randint(0, num_cells, size=num_users) for _ in range(population_size)]
#         self.loudness = [1.0 for _ in range(population_size)]
#         self.pulse_rate = [0.5 for _ in range(population_size)]
#         self.best = max(self.population, key=self.fitness)
    
#     def fitness(self, solution):
#         return evaluate_detailed_solution(self.env, solution)["fitness"]
    
#     def optimize(self):
#         for t in range(self.iterations):
#             for i in range(self.population_size):
#                 # Use the seeded RNG to generate frequency
#                 freq = self.freq_min + (self.freq_max - self.freq_min) * self.rng.rand()
#                 new_solution = self.population[i].copy()
#                 for j in range(self.num_users):
#                     # Use seeded RNG for decision making
#                     if self.rng.rand() > self.pulse_rate[i]:
#                         new_solution[j] = self.rng.randint(0, self.num_cells)
#                 if self.fitness(new_solution) > self.fitness(self.population[i]) and self.rng.rand() < self.loudness[i]:
#                     self.population[i] = new_solution
#                     self.loudness[i] *= self.alpha
#                     self.pulse_rate[i] = self.pulse_rate[i] * (1 - np.exp(-self.gamma * t))
#                 if self.fitness(self.population[i]) > self.fitness(self.best):
#                     self.best = self.population[i]
#         return self.best
