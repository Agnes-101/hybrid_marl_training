# algorithms/cs.py
import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class CSOptimization:
    def __init__(self, seed=None):
        """Cuckoo Search with hardcoded parameters for 6G optimization.
        
        Decoupled Initialization: Self-configures with hardcoded parameters.
        """
        # Hardcoded optimization parameters
        self.colony_size = 30
        self.iterations = 50
        self.pa = 0.25  # Abandonment probability
        self.levy_scale = 1.0  # Scale factor for levy flights
        self.seed = seed if seed is not None else np.random.randint(0, 10000)
        
        # Visualization states
        self.positions = np.empty((0, 2))  # (solution_diversity, fitness)
        self.fitness_history = []
        self.population = None
        
        # Initialize the RNG for reproducibility
        self.rng = np.random.RandomState(self.seed)
        
        # Environment will be stored during run()
        self.env = None

    def run(self, env: NetworkEnvironment) -> dict:
        """Unified entry point for hybrid training system.
        
        Returns:
            dict: A dictionary containing the best solution, its metrics, 
                  and visualization data (positions and fitness history).
        """
        self.env = env  # Store environment for use in visualization updates
        num_ue = env.num_ue
        num_bs = env.num_bs
        
        # Initialize population
        self.population = [
            self.rng.randint(0, num_bs, num_ue)
            for _ in range(self.colony_size)
        ]
        
        best_solution = None
        best_fitness = -np.inf

        for iteration in range(self.iterations):
            # Generate new solutions using levy flights
            for i in range(self.colony_size):
                new_sol = self._levy_flight_update(self.population[i], num_bs)
                new_fitness = env.evaluate_detailed_solution(new_sol)["fitness"]
                
                # Evaluate current fitness of the solution
                current_fitness = env.evaluate_detailed_solution(self.population[i])["fitness"]
                # Replace current solution if new one is better
                if new_fitness > current_fitness:
                    self.population[i] = new_sol
                    # Update the global best if applicable
                    if new_fitness > best_fitness:
                        best_fitness = new_fitness
                        best_solution = new_sol.copy()

            # Abandon worst solutions based on the abandonment probability
            self._abandon_worst_solutions(env, num_bs)
            
            # Update visualization states for backward compatibility
            self._update_visualization(best_solution, iteration)

        return {
            "solution": best_solution,
            "metrics": env.evaluate_detailed_solution(best_solution),
            "agents": {
                "positions": self.positions,
                "fitness": self.fitness_history
            }
        }

    def _levy_flight_update(self, solution: np.ndarray, num_bs: int) -> np.ndarray:
        """Generate new solution using Levy flight characteristics."""
        # Generate a random step following a normal distribution scaled by levy_scale
        step = self.rng.normal(0, self.levy_scale, size=solution.shape)
        new_sol = solution + np.round(step).astype(int)
        # Ensure the new solution is within valid base station indices
        return np.clip(new_sol % num_bs, 0, num_bs - 1)

    def _abandon_worst_solutions(self, env: NetworkEnvironment, num_bs: int):
        """Replace a fraction of the worst solutions with random ones."""
        fitnesses = [env.evaluate_detailed_solution(sol)["fitness"] for sol in self.population]
        # Identify indices of the worst solutions (lowest fitness)
        worst_count = int(self.pa * self.colony_size)
        worst_idx = np.argsort(fitnesses)[:worst_count]
        
        for idx in worst_idx:
            self.population[idx] = self.rng.randint(0, num_bs, size=env.num_ue)

    def _update_visualization(self, best_solution: np.ndarray, iteration: int):
        """Track solution diversity and fitness for visualization.
        
        This method converts the best solution into a 2D coordinate using its 
        diversity (normalized count of unique base stations) and fitness.
        """
        if best_solution is None:
            return  # Skip update if no best solution has been found yet
        
        # Calculate diversity: normalized count of unique base stations in best_solution
        best_diversity = len(np.unique(best_solution)) / self.env.num_bs
        current_fitness = self.env.evaluate_detailed_solution(best_solution)["fitness"]
        
        # Append current state as 2D coordinate: (diversity, fitness)
        self.positions = np.vstack([self.positions, [best_diversity, current_fitness]])
        self.fitness_history.append(current_fitness)
        # Optionally, print debugging information
        print(f"Iteration {iteration}: Diversity = {best_diversity:.2f}, Fitness = {current_fitness:.4f}")



# import numpy as np
# import random
# from envs.custom_channel_env import evaluate_detailed_solution

# class CSOptimization:
#     def __init__(self, num_users, num_cells, env, colony_size=30, iterations=50, pa=0.25, seed=None):
#         self.num_users = num_users
#         self.num_cells = num_cells
#         self.env = env
#         self.colony_size = colony_size
#         self.iterations = iterations
#         self.pa = pa
#         self.seed = seed
#         self.rng = np.random.RandomState(seed)
#         # Initialize population using the seeded RNG
#         self.population = [self.rng.randint(0, num_cells, size=num_users) for _ in range(colony_size)]
    
#     def fitness(self, solution):
#         return evaluate_detailed_solution(self.env, solution)["fitness"]
    
#     def levy_flight(self):
#         # Generate a levy flight step using the seeded RNG
#         return self.rng.randint(-1, 2, size=self.env.num_users)
    
#     def optimize(self):
#         for _ in range(self.iterations):
#             for i in range(self.colony_size):
#                 new_solution = self.population[i].copy()
#                 step = self.levy_flight()
#                 new_solution = (new_solution + step) % self.env.num_cells
#                 if self.fitness(new_solution) > self.fitness(self.population[i]):
#                     self.population[i] = new_solution
#             fitnesses = [self.fitness(sol) for sol in self.population]
#             indices = np.argsort(fitnesses)
#             num_abandon = int(self.pa * self.colony_size)
#             for idx in indices[:num_abandon]:
#                 self.population[idx] = self.rng.randint(0, self.env.num_cells, size=self.env.num_users)
#         return max(self.population, key=self.fitness)
