# # algorithms/pfo.py

import numpy as np
import torch
from envs.custom_channel_env import NetworkEnvironment

class PolarFoxOptimization:
    def __init__(self, env, kpi_logger=None):
                
        self.env = env
        self.num_users = env.num_ue
        self.num_cells = env.num_bs
        
        self.population_size = 40
        self.iterations = 50
        self.mutation_factor = 0.2
        self.jump_rate = 0.2 
        self.follow_rate = 0.3  
        self.seed = 42
        np.random.seed(self.seed)
        self.kpi_logger = kpi_logger
        
        # Group parameters [PF, LF, a, b, m]
        self.types = np.array([
            [2, 2, 0.9, 0.9, 0.1],   # Group 0: Balanced
            [10, 2, 0.2, 0.9, 0.3],  # Group 1: Explorer
            [2, 10, 0.9, 0.2, 0.3],  # Group 2: Follower
            [2, 12, 0.9, 0.9, 0.01]  # Group 3: Conservative
        ])
        
        # Initialize population with groups
        self.population = self.initialize_population(env)
        self.group_weights = [1000, 1000, 1000, 1000]
        self.best_solution = None
        self.best_fitness = -np.inf
        # For live updates, you can also keep track of positions and fitness history:
        # self.positions = None  # will be computed from population later        
        self.positions = np.empty((0, 3))  # Match DE's 3D position format
        self.fitness = np.full(self.iterations, np.nan)  # Pre-allocate fitness array
        self.best_fitness_history = []  # Rename from historical_bests
        self.best_metrics_history = []  # To store metrics per iteration
        self._rng = np.random.RandomState(self.seed)  # DE-style RNG
        
    def initialize_population(self, env: NetworkEnvironment):
        population = []
        for _ in range(self.population_size):
            if np.random.rand() < 0.2:
                # Heuristic: Assign users to nearest cell
                fox = np.array([self.find_nearest_cell(ue.position, env) for ue in env.ues])
            else:
                fox = np.random.randint(0, self.num_cells, size=self.num_users)
            population.append(fox)
        # print(f"Population initialized: {population}")
        return population


    # def find_nearest_cell(self, user_position, env: NetworkEnvironment):
    #     cell_positions = np.array([bs.position for bs in env.base_stations])
    #     distances = np.linalg.norm(cell_positions - user_position, axis=1)
    #     return np.argmin(distances)    

    def find_nearest_cell(self, user_position, env: NetworkEnvironment):
        if not isinstance(user_position, torch.Tensor):
            user_position = torch.tensor(user_position, dtype=torch.float32)
        cell_positions = torch.stack([bs.position for bs in env.base_stations]).to(user_position.device)
        with torch.no_grad():
            distances = torch.norm(cell_positions - user_position, dim=1)

        # distances = torch.norm(cell_positions - user_position, dim=1)
        return torch.argmin(distances).item()

    
    def _find_alternative_bs(self, user_indices, counts):
        capacities = np.array([bs.capacity for bs in self.env.base_stations])
        available_cells = np.where(counts < capacities)[0]
        if len(available_cells) == 0:
            return np.random.randint(0, self.num_cells, size=len(user_indices))
        new_assignment = available_cells[np.argmin(counts[available_cells])]
        return new_assignment
    
    def calculate_group_distribution(self):
        """Distribute foxes according to initial group ratios"""
        base_dist = np.array([0.25, 0.25, 0.25, 0.25])
        counts = np.floor(base_dist * self.population_size).astype(int)
        counts[3] = self.population_size - sum(counts[:3])
        return counts

    def create_fox(self, group_id):
        """Create fox with group-specific parameters"""
        fox = {
            "solution": self.generate_initial_solution(),
            "PF": self.types[group_id, 0],
            "LF": self.types[group_id, 1],
            "a": self.types[group_id, 2],
            "b": self.types[group_id, 3],
            "m": self.types[group_id, 4],
            "group": group_id
        }
        return fox

    # algorithms/pfo.py (modified)
    def generate_initial_solution(self):
        """20% use nearest-cell heuristic"""
        if np.random.rand() < 0.2:
            # Replace self.env.users with self.env.user_positions
            return np.array([self.env.find_nearest_cell(pos) for pos in self.env.user_positions])
        else:
            return np.random.randint(0, self.num_cells, size=self.num_users)

    def compute_fitness(self, solution):
        return self.env.evaluate_detailed_solution(solution)["fitness"]

    def adaptive_parameters(self, iteration):
        """Decay jump power (a), boost follow power (b) over time"""
        decay_factor = 1 - (iteration / self.iterations) ** 0.5
        for i in range(4):
            self.types[i, 2] = max(self.types[i, 2] * decay_factor, 0.1)
            self.types[i, 3] = min(self.types[i, 3] / decay_factor, 0.99)

    def jump_experience(self, fox):
        """Randomly jump to explore new solutions."""
        new_fox = fox.copy()
        num_jumps = int(self.num_users * self.jump_rate)  # Convert to integer
        if num_jumps < 1:  # Ensure at least 1 jump
            num_jumps = 1
        indices = np.random.choice(self.num_users, num_jumps, replace=False)
        new_fox[indices] = np.random.randint(0, self.num_cells, size=num_jumps)
        return new_fox

    def follow_leader(self, current_fox, best_fox):
        """Update part of the solution to follow the best solution."""
        new_fox = current_fox.copy()
        num_follow = int(self.num_users * self.follow_rate)  # <-- FIXED
        indices = np.random.choice(self.num_users, num_follow, replace=False)
        new_fox[indices] = best_fox[indices]
        return new_fox

    # def repair(self, solution):
    #     """Ensure cell capacity constraints"""
    #     cell_counts = np.bincount(solution, minlength=self.num_cells)
    #     overloaded = np.where(cell_counts > self.env.cell_capacity)[0]
        
    #     for cell in overloaded:
    #         users = np.where(solution == cell)[0]
    #         for user in users[self.env.cell_capacity:]:
    #             solution[user] = np.argmin(cell_counts)
    #             cell_counts[solution[user]] += 1
    #     return solution
    
    def repair(self, solution):
        """DE-style vectorized repair"""
        counts = np.bincount(solution, minlength=self.num_cells)
        capacities = np.array([bs.capacity for bs in self.env.base_stations])
        
        overloaded = np.where(counts > capacities)[0]
        for bs_id in overloaded:
            excess = counts[bs_id] - capacities[bs_id]
            users = np.where(solution == bs_id)[0][:excess]
            solution[users] = self._find_alternative_bs(users, counts)
        
        return solution
    
    def leader_motivation(self, stagnation_count):
        """Reset underperforming foxes and adjust groups"""
        num_mutation = int(self.population_size * self.mutation_factor)
        for i in range(num_mutation):
            group_id = np.random.choice(4, p=self.group_weights/np.sum(self.group_weights))
            self.population[i] = self.create_fox(group_id)
        
        # Boost weights of best-performing group
        if self.best_solution is not None:
            best_group = self.population[np.argmax([f["group"] for f in self.population])]["group"]
            self.group_weights[best_group] += stagnation_count * 100
            
    def _calculate_visual_positions(self):
        """DE-style position projection"""
        visual_positions = []
        for solution in self.population:
            # Feature 1: Load balance
            counts = np.bincount(solution, minlength=self.num_cells)
            x = np.std(counts)
            
            # Feature 2: Average SINR (via temporary env state)
            with self.env.temporary_state():
                self.env.apply_solution(solution)
                y = np.mean([ue.sinr for ue in self.env.users])
            
            visual_positions.append([x, y, self.fitness(solution)])
        self.positions = np.array(visual_positions)    

    
    def run(self, env: NetworkEnvironment, visualize_callback: callable = None, kpi_logger=None) -> np.ndarray:
        """Enhanced optimization loop with anti-stagnation mechanisms."""
        best_solution = self.population[0].copy()
        best_fitness = -np.inf
        historical_bests = []
        no_improvement_streak = 0
        stagnation_threshold = 5  # Increased from 3
        diversity_window = 10  # Track diversity over last N iterations
        mutation_reset = 0.1  # Minimum mutation factor
        
        # Initialize population diversity tracking
        diversity_history = []
        # current_metrics = env.evaluate_detailed_solution(current_solution)
        best_iter_metrics = env.evaluate_detailed_solution(best_solution)
        
        for iteration in range(self.iterations):
            # Adaptive parameter update
            self.adaptive_parameters(iteration)
            
            # Evaluate population
            fitness_values = [self.compute_fitness(fox) for fox in self.population]
            
            # Update best solution with elitism
            current_best_idx = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]
            current_best_solution = self.population[current_best_idx].copy()
            current_best_metrics = env.evaluate_detailed_solution(current_best_solution)

            
            # Maintain diversity tracking
            diversity = np.std(fitness_values)
            diversity_history.append(diversity)
            if len(diversity_history) > diversity_window:
                diversity_history.pop(0)

            # Update best solution with momentum (prevent oscillation)
            if current_best_fitness > best_fitness * 1.001:  # 0.1% improvement threshold
                best_fitness = current_best_fitness
                best_solution = self.population[current_best_idx].copy()
                no_improvement_streak = 0
            else:
                no_improvement_streak += 1

            # Log metrics per iteration via KPI logger if available
            if self.kpi_logger:
                self.kpi_logger.log_metrics(
                    episode=iteration,
                    phase="metaheuristic",
                    algorithm="pfo",
                    metrics=current_best_metrics # Log full metrics {"fitness": best_fitness, "diversity": diversity}
                )
                print(f"PFO Iter {iteration}: Best Fitness = {best_fitness:.4f}, Diversity = {diversity:.2f}")
            
            historical_bests.append(best_fitness)
            
            # # Periodic live dashboard updates (every 5 iterations)
            # if visualize_callback and iteration % 5 == 0:
            #     # Compute visualization positions (example: use diversity as x and best fitness as y)
            #     positions = np.column_stack((
            #         np.full((self.population_size,), diversity),   # Dummy x: diversity for all
            #         np.array(fitness_values)                        # y: fitness values
            #     ))
            #     self.positions = positions  # Save current positions for visualization
                
            #     visualize_callback({
            #         "positions": positions.tolist(),
            #         "fitness": self.best_fitness_history,
            #         "algorithm": "pfo",
            #         "env_state": self.env.get_current_state()
            #     })
            #     print(f"PFO Visual Update @ Iter {iteration}")
                
            # Update global best solution
            if current_best_metrics["fitness"] > self.best_fitness:
                self.best_fitness = current_best_metrics["fitness"]
                best_solution = current_best_solution.copy()
                
            # Enhanced stagnation detection with diversity check
            avg_diversity = np.mean(diversity_history[-3:]) if diversity_history else 0
            if (iteration > 20 and 
                no_improvement_streak >= stagnation_threshold and 
                avg_diversity < 0.5 * np.mean(diversity_history)):
                
                # Aggressive mutation boost
                self.mutation_factor = min(1.0, self.mutation_factor * 2)
                no_improvement_streak = max(0, no_improvement_streak - 2)
                
                # Diversity injection
                num_replace = int(0.2 * self.population_size)
                for i in range(num_replace):
                    self.population[-(i+1)] = self.random_solution()
                
                print(f"Iter {iteration}: Mutation ↑ {self.mutation_factor:.2f}, Diversity injection")

            # Dynamic population management
            sorted_indices = np.argsort(fitness_values)[::-1]
            
            # Keep top 10% elites unchanged
            elite_count = max(1, int(0.1 * self.population_size))
            elites = [self.population[i].copy() for i in sorted_indices[:elite_count]]
            
            # Generate new population
            new_population = elites.copy()
            
            # Create remaining population through enhanced operations
            while len(new_population) < self.population_size:
                parent = self.population[np.random.choice(sorted_indices[:elite_count*2])]
                
                if np.random.rand() < 0.7:  # Favor exploitation
                    child = self.follow_leader(parent, best_solution)
                else:  # Exploration
                    child = self.jump_experience(parent)
                    
                # Apply mutation with adaptive probability
                mutation_prob = 0.3 + (0.5 * (self.mutation_factor / 1.0))
                if np.random.rand() < mutation_prob:
                    child = self.jump_experience(child)
                    
                new_population.append(child)

            self.population = new_population
            
            # Adaptive mutation decay
            if no_improvement_streak == 0 and self.mutation_factor > mutation_reset:
                self.mutation_factor *= 0.95  # Gradual decay on improvement
                
            # Update environment with current best solution
            self.best_solution = best_solution.copy()
            self.env.apply_solution(self.best_solution)
            actions = {
                f"bs_{bs_id}": np.where(self.best_solution == bs_id)[0].tolist()
                for bs_id in range(self.env.num_bs)
            }
            self.env.step(actions)  # Update environment state
            
            # Progress tracking
            historical_bests.append(best_fitness)
            print(f"Iter {iteration+1}: Best = {best_fitness:.4f}, "
                f"Mutation = {self.mutation_factor:.2f}, "
                f"Diversity = {diversity:.2f}")

        # return best_solution
        # Return DE-style output
        return {
            "solution": self.best_solution,
            "metrics": best_iter_metrics,
            "agents": {
                "positions": self.positions.tolist(),
                "fitness": self.fitness.tolist(),
                "algorithm": "pfo"
            }
        }
    