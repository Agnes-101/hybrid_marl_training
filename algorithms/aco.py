# algorithms/aco.py
import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class ACO:
    def __init__(self, env, kpi_logger=None):
        """Ant Colony Optimization with adaptive parameters and vectorized solution construction"""
        # Optimization parameters
        self.env=env
        self.ants = 30
        self.iterations = 50
        self.evaporation_rate = 0.1
        self.alpha_init = 1    # initial pheromone influence
        self.beta_init = 2     # initial heuristic influence (if applicable)
        self.seed = 42 # seed if seed is not None else np.random.randint(0, 10000)
        
        # Visualization states
        self.positions = np.empty((0, 2))  # Stores (mean, std) of pheromones per iteration
        self.fitness_history = []
        self.pheromones = None
        self.kpi_logger = kpi_logger
        # Initialize the random generator for reproducibility
        self.rng = np.random.RandomState(self.seed)

    def run(self, env: NetworkEnvironment, visualize_callback: callable = None, kpi_logger=None) -> dict: 
        """Main interface for hybrid training system"""
        num_ue = env.num_ue
        num_bs = env.num_bs
        
        # Initialize pheromones with a base value and a bit of noise
        self.pheromones = np.ones((num_ue, num_bs)) * 0.1
        self.pheromones += self.rng.uniform(0, 0.01, (num_ue, num_bs))
        
        self.best_solution = None
        best_fitness = -np.inf

        for iteration in range(self.iterations):
            # Adaptive adjustment of alpha and beta over iterations
            alpha = self.alpha_init + (iteration / self.iterations) * 2   # increasing pheromone influence
            beta = self.beta_init - (iteration / self.iterations) * 1.5     # decreasing heuristic influence

            # Generate solutions using a vectorized method
            solutions = self._construct_solution_vectorized(num_bs, alpha, beta)
            
            # Evaluate each solution's fitness
            fitness_values = np.array([env.evaluate_detailed_solution(sol)["fitness"] for sol in solutions])
            current_best_idx = np.argmax(fitness_values)
            
            # Update the best solution if a better one is found
            if fitness_values[current_best_idx] > best_fitness:
                best_fitness = fitness_values[current_best_idx]
                self.best_solution = solutions[current_best_idx].copy()

            # Update pheromones using normalized fitness contributions
            self._update_pheromones(solutions, fitness_values)
            self._update_visualization(env, iteration)
            
            # After evaluating fitness_values:
            current_best_metrics = env.evaluate_detailed_solution(solutions[current_best_idx])
            self.fitness_history.append(current_best_metrics["fitness"])

            # ✅ DE/PFO-style logging
            if self.kpi_logger:
                self.kpi_logger.log_metrics(
                    episode=iteration,
                    phase="metaheuristic",
                    algorithm="aco",
                    metrics=current_best_metrics  # Log full metrics, not just fitness
                )
                print(f"ACO Iter {iteration}: Best Fitness = {best_fitness:.4f}")
                
            # # ✅ Visualization trigger (every 5 iterations)
            # if visualize_callback and iteration % 5 == 0:
            #     visualize_callback({
            #         "positions": self.positions.tolist(),
            #         "fitness": self.fitness_history,
            #         "algorithm": "aco",
            #         "env_state": env.get_current_state()
            #     })
            
            # Mirror DE's environment interaction
            env.apply_solution(self.best_solution)
            actions = {
                f"bs_{bs_id}": np.where(self.best_solution == bs_id)[0].tolist()
                for bs_id in range(env.num_bs)
            }
            env.step(actions)  # Update network state
            
        return {
            "solution": self.best_solution,
            "metrics": env.evaluate_detailed_solution(self.best_solution),
            "agents": {
                "positions": self.positions.tolist(),
                "fitness": self.fitness_history,
                "algorithm": "aco"
            }
        }

    def _construct_solution_vectorized(self, num_bs: int, alpha: float, beta: float) -> np.ndarray:
        """
        Generate solutions for all ants in a vectorized way using pheromone probabilities.
        The 'beta' parameter is available for future heuristic incorporation if needed.
        """
        # Calculate probabilities: pheromones raised to alpha
        probs = self.pheromones ** alpha
        
        # Normalize probabilities for each user
        row_sums = probs.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        probs = probs / row_sums
        
        # Number of users
        num_users = self.pheromones.shape[0]
        # Create an empty array to store solutions for all ants
        solutions = np.empty((self.ants, num_users), dtype=int)
        # For each user, sample base stations for all ants based on computed probabilities
        for i in range(num_users):
            solutions[:, i] = self.rng.choice(num_bs, size=self.ants, p=probs[i])
        return solutions

    def _update_pheromones(self, solutions: np.ndarray, fitness_values: np.ndarray):
        """Update pheromones with evaporation and fitness-based reinforcement."""
        # Evaporation: reduce all pheromones
        self.pheromones *= (1 - self.evaporation_rate)
        
        # Normalize fitness contributions to avoid excessive updates
        max_fitness = np.max(fitness_values) if np.max(fitness_values) > 0 else 1
        
        # Reinforce pheromones based on fitness; iterate over each solution and its fitness
        for sol, fitness in zip(solutions, fitness_values):
            for user, bs in enumerate(sol):
                self.pheromones[user, bs] += fitness / max_fitness  # normalized update
                
        # Clip pheromone values to maintain numerical stability
        self.pheromones = np.clip(self.pheromones, 1e-10, 1e5)

    def _update_visualization(self, env: NetworkEnvironment, iteration: int):
        """Update visualization states: pheromone statistics and fitness history."""
        # Compute mean and standard deviation of pheromones for each base station
        mean_pheromones = self.pheromones.mean(axis=0)
        std_pheromones = self.pheromones.std(axis=0)
        
        # Store these statistics as 2D coordinates (X = mean, Y = std)
        self.positions = np.vstack([
            self.positions,
            np.column_stack([mean_pheromones, std_pheromones])
        ])
        
        # Log the current best fitness for visualization purposes
        self.fitness_history.append(
            env.evaluate_detailed_solution(self.best_solution)["fitness"]
        )


# # algorithms/aco.py
# import numpy as np
# from envs.custom_channel_env import evaluate_detailed_solution

# class ACOOptimization:
#     def __init__(self, num_users, num_cells, env, ants=30, iterations=50, 
#                 evaporation_rate=0.1, alpha=1, beta=2, seed=None):
#         self.num_users = num_users
#         self.num_cells = num_cells
#         self.env = env
#         self.ants = ants
#         self.iterations = iterations
#         self.evaporation_rate = evaporation_rate
#         self.alpha = alpha
#         self.beta = beta
#         self.seed = seed
#         self.rng = np.random.RandomState(seed)
        
#         # Initialize pheromones with safe values
#         self.pheromones = np.ones((num_users, num_cells)) * 0.1
#         self.pheromones += self.rng.uniform(0, 0.01, (num_users, num_cells))
#         self.best_solution = None
#         self.best_fitness = -np.inf

#     def fitness(self, solution):
#         return max(evaluate_detailed_solution(self.env, solution)["fitness"], 0)

#     def construct_solution(self):
#         solution = np.zeros(self.num_users, dtype=int)
#         for user in range(self.num_users):
#             # Numerical stability using log-exp trick
#             with np.errstate(divide='ignore'):
#                 log_pher = np.log(self.pheromones[user] + 1e-20)
            
#             log_prob = self.alpha * log_pher
#             log_prob -= np.max(log_prob)  # Prevent overflow
#             probabilities = np.exp(log_prob) + 1e-10
#             probabilities /= probabilities.sum()
            
#             solution[user] = self.rng.choice(self.num_cells, p=probabilities)
#         return solution

#     def update_pheromones(self, solutions):
#         # Apply evaporation and ensure positivity
#         self.pheromones = np.clip(self.pheromones * (1 - self.evaporation_rate), 1e-10, None)
        
#         # Only allow positive pheromone additions
#         for sol in solutions:
#             fit = self.fitness(sol)
#             for user in range(self.num_users):
#                 self.pheromones[user, sol[user]] += fit
                
#         # Post-update clipping
#         self.pheromones = np.clip(self.pheromones, 1e-10, 1e5)

#     def optimize(self):
#         for iteration in range(self.iterations):
#             solutions = [self.construct_solution() for _ in range(self.ants)]
#             current_best = max(solutions, key=lambda x: self.fitness(x))
            
#             if self.fitness(current_best) > self.best_fitness:
#                 self.best_fitness = self.fitness(current_best)
#                 self.best_solution = current_best.copy()
            
#             # Update with both current and historical best
#             self.update_pheromones(solutions + [self.best_solution])
            
#             print(f"Iter {iteration+1}: Best Fitness = {self.best_fitness:.4f}")

#         return self.best_solution

