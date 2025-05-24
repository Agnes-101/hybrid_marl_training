
# import numpy as np
# from envs.custom_channel_env import NetworkEnvironment 
# from concurrent.futures import ThreadPoolExecutor
# import copy


# class PolarFoxOptimization:
#     def __init__(self, env: NetworkEnvironment, iterations=20, population_size=60,
#                 mutation_factor=0.2, group_weights: list[float] | None = None, kpi_logger=None):
#         self.env = env
#         self.num_users = env.num_ue
#         self.num_cells = env.num_bs
#         self.population_size = population_size
#         self.iterations = iterations
#         self.mutation_factor = mutation_factor
#         self.seed = 42
#         np.random.seed(self.seed)
#         self.kpi_logger = kpi_logger

#         # PFOA CORE: Group parameters [PF0, LF0, a, b, m]
#         self.types = np.array([
#             [2, 2, 0.9, 0.9, 0.1],   # Group 0
#             [10, 2, 0.2, 0.9, 0.3],  # Group 1
#             [2, 10, 0.9, 0.2, 0.3],  # Group 2
#             [2, 12, 0.9, 0.9, 0.01]  # Group 3
#         ])
        
#         # PFOA CORE: Initialize population with group structures
#         self.population = []
#         counts = np.floor(0.25 * np.ones(4) * population_size).astype(int)
#         counts[3] = population_size - counts[:3].sum()
#         for k in range(4):
#             for _ in range(counts[k]):
#                 fox = {
#                     'solution': self.generate_initial_solution(),
#                     'fitness': None,
#                     'PF': self.types[k,0], 'LF': self.types[k,1],
#                     'a': self.types[k,2], 'b': self.types[k,3], 'm': self.types[k,4],
#                     'group': k
#                 }
#                 self.population.append(fox)
                
#         # PFOA CORE: Weight initialization
#         self.W = np.array([1000.0]*4)  # Original PFOA weights
#         self.best_solution = None
#         self.best_fitness = -np.inf
#         self.stagnation_count = 0
#         self.executor = ThreadPoolExecutor(max_workers=4)
        
    
#     # def compute_fitness_batch(self, solutions):
#     #     # Don’t re-enter as a context manager—just map over it:
#     #     print(f"Computing fitness batch on {len(solutions)} sols")
        
#     #     for sol in solutions:
#     #         print(f"Solution length: {len(sol)}")
#     #         print(f"BS indices out of range: {sol.min()},{sol.max()}")
#     #         # Must be exactly one BS index per UE
#     #         assert len(sol) == self.num_users, (
#     #             f"Solution length {len(sol)} != num_users {self.num_users}"
#     #         )
#     #         # BS indices must be in [0, num_cells)
#     #         assert sol.min() >= 0 and sol.max() < self.num_cells, (
#     #             f"BS indices out of range: {sol.min()}–{sol.max()}"
#     #         )
#     #     print("ALmost..")
#     #     results = list(self.executor.map(self.env.evaluate_detailed_solution, solutions))
#     #     return [(r['fitness'], np.mean(r.get('sinr', 0))) for r in results]
    
#     def compute_fitness_batch(self, solutions):
#         results = []
#         def evaluate_with_env_copy(solution):
#             # Create a deep copy of the environment for each evaluation
#             env_copy = copy.deepcopy(self.env)
#             return env_copy.evaluate_detailed_solution(solution)
        
#         with ThreadPoolExecutor(max_workers=4) as executor:
#             results = list(executor.map(evaluate_with_env_copy, solutions))
        
#         return results
    
#     def vectorized_repair(self, solutions):
#         """Vectorized repair for multiple solutions"""
#         repaired = []
#         for sol in solutions:
#             counts = np.bincount(sol, minlength=self.num_cells)
#             capacities = np.array([int(bs.capacity) for bs in self.env.base_stations])
            
#             # Find all overloaded users at once
#             overload_mask = np.isin(sol, np.where(counts > capacities)[0])
#             overloaded_users = np.where(overload_mask)[0]
            
#             if len(overloaded_users) > 0:
#                 # Batch find alternatives
#                 alt_bs = self._find_alternative_bs(overloaded_users, counts)
#                 sol[overloaded_users] = alt_bs
                
#             repaired.append(sol)
#         return repaired
    
#     def batch_evaluate_population(self):
#         """Batch evaluate all solutions with detailed results"""
#         print("Starting Batch Evaluation...")
#         solutions = [f['solution'] for f in self.population]
#         results = self.compute_fitness_batch(solutions)
        
#         for fox, result in zip(self.population, results):
#             fox['fitness'] = result['fitness']
#             fox['sinr'] = result.get('average_sinr', 0)  # Use get() with default in case it's missing
#         print("Finished Batch evaluation..")
#     # PFOA CORE: Revised experience phase with PF decay
#     def experience_phase(self, fox, max_tries=10):
#         s = fox['solution']
#         current_fitness = self.compute_fitness(s)
#         PF, a, m = fox['PF'], fox['a'], fox['m']
#         PF0 = self.types[fox['group'], 0]
        
#         for _ in range(max_tries):
#             # Generate new solution using current PF
#             flips = np.random.rand(self.num_users) < (PF / PF0)
#             new_solution = s.copy()
#             new_solution[flips] = np.random.randint(0, self.num_cells, flips.sum())
#             new_solution = self.repair(new_solution)
#             new_fitness = self.compute_fitness(new_solution)
            
#             PF *= a  # Decay power factor
#             if new_fitness < current_fitness or PF < m * PF0:
#                 break
                
#         if new_fitness < current_fitness:
#             fox.update({
#                 'solution': new_solution,
#                 'fitness': new_fitness,
#                 'PF': PF
#             })

#     # PFOA CORE: Revised leader phase with LF decay
#     def leader_phase(self, fox, leader_solution, max_tries=10):
#         s = fox['solution']
#         current_fitness = fox['fitness'] or self.compute_fitness(s)
#         LF, b, m = fox['LF'], fox['b'], fox['m']
#         LF0 = self.types[fox['group'], 1]
        
#         for _ in range(max_tries):
#             # Follow leader using current LF
#             copies = np.random.rand(self.num_users) < (LF / LF0)
#             new_solution = s.copy()
#             new_solution[copies] = leader_solution[copies]
#             new_solution = self.repair(new_solution)
#             new_fitness = self.compute_fitness(new_solution)
            
#             LF *= b  # Decay leadership factor
#             if new_fitness < current_fitness or LF < m * LF0:
#                 break
                
#         if new_fitness < current_fitness:
#             fox.update({
#                 'solution': new_solution,
#                 'fitness': new_fitness,
#                 'LF': LF
#             })

#     # PFOA CORE: Stagnation-aware mutation
#     def selective_mutation(self):
#         print("Mutation Phase.....")
#         if self.stagnation_count > 50:  # Original PFOA condition
#             num_mutate = self.population_size - 1  # Mutate all except leader
#         else:
#             num_mutate = int(self.mutation_factor * self.population_size)
            
#         worst = sorted(self.population, key=lambda f: f['fitness'])[:num_mutate]
#         for fox in worst:
#             fox['solution'] = self.generate_initial_solution()
#             fox['fitness'] = self.compute_fitness(fox['solution'])

#     # PFOA CORE: Weight update formula W_k += I²/NG_k
#     def update_group_weights(self, iteration):
#         print("Updating Group weights...")
#         group_counts = np.array([sum(f['group']==k for f in self.population) for k in range(4)])
#         group_counts = np.clip(group_counts, 1, None)  # Avoid division by zero
#         self.W += (iteration ** 2) / group_counts
#         self.W /= self.W.sum()  # Normalize for probability sampling

#     # PFOA CORE: Fatigue simulation from original algorithm
#     def fatigue_update(self):
#         print("Fatigue updates....")
#         group_counts = [sum(f['group']==k for f in self.population) for k in range(4)]
#         for k in range(4):
#             if group_counts[k] < 5:
#                 # Boost parameters for rare groups
#                 self.types[k, 2:] = [0.99, 0.99, 0.001]
#             else:
#                 # Decay parameters for common groups
#                 self.types[k, 2] = max(self.types[k,2] - 0.001, 0.9)  # a
#                 self.types[k, 3] = max(self.types[k,3] - 0.001, 0.9)  # b
#                 self.types[k, 4] = min(self.types[k,4] + 0.0001, 0.01) # m

#     # Maintained network-specific methods
#     def generate_initial_solution(self):
#         if np.random.rand() < 0.4:
#             return np.array([self.find_nearest_cell(ue.position) for ue in self.env.ues])
#         return np.random.randint(0, self.num_cells, self.num_users)

#     def find_nearest_cell(self, position):
#         cell_positions = np.stack([bs.position for bs in self.env.base_stations])
#         return np.argmin(np.linalg.norm(cell_positions - position, axis=1))
    
#     def _find_alternative_bs(self, user_indices, counts):
#         capacities = np.array([bs.capacity for bs in self.env.base_stations])
#         available_cells = np.where(counts < capacities)[0]
        
#         if len(available_cells) == 0:
#             # Handle scalar/array input for user_indices
#             if np.isscalar(user_indices):
#                 return np.random.randint(0, self.num_cells)
#             else:
#                 return np.random.randint(0, self.num_cells, size=len(user_indices))
        
#         # Find least-loaded available cell
#         least_loaded = available_cells[np.argmin(counts[available_cells])]
        
#         # Return same cell for all users in indices
#         if np.isscalar(user_indices):
#             return least_loaded
#         else:
#             return np.full(len(user_indices), least_loaded)

#     def repair(self, solution):
#         """Vectorized repair with integer casting"""
#         counts = np.bincount(solution, minlength=self.num_cells)
#         capacities = np.array([int(bs.capacity) for bs in self.env.base_stations])  # Ensure integer
        
#         overloaded = np.where(counts > capacities)[0]
#         for bs_id in overloaded:
#             excess = counts[bs_id] - capacities[bs_id]
#             users = np.where(solution == bs_id)[0]
#             # Ensure we only move excess users using integer slicing
#             for u in users[capacities[bs_id]:capacities[bs_id] + excess]:  # ✅ Integer bounds
#                 solution[u] = self._find_alternative_bs(u, counts)
        
#         return solution
    
#     def _calculate_visual_positions(self):
#         """DE-style position projection (updated for PFOA structure)"""
#         visual_positions = []
#         for fox in self.population:  # Iterate over foxes, not raw solutions
#             solution = fox["solution"]
#             fitness = fox["fitness"]  # Use precomputed fitness
            
#             # Feature 1: Load balance (same as before)
#             counts = np.bincount(solution, minlength=self.num_cells)
#             x = np.std(counts)
            
#             # Feature 2: Average SINR (with temporary state safety)
#             with self.env.temporary_state():
#                 self.env.apply_solution(solution)
#                 y = np.mean([ue.sinr for ue in self.env.users])
            
#             visual_positions.append([x, y, fitness])  # Use stored fitness
        
#         self.positions = np.array(visual_positions)
        
#     def compute_fitness(self, solution):
#         return self.env.evaluate_detailed_solution(solution)['fitness']
#     def batch_experience_phase(self, max_tries=10):
#         """Batch implementation of the experience phase for all foxes"""
#         print("Starting batch experience phase...")
        
#         # Prepare batches of solutions to evaluate
#         all_new_solutions = []
#         original_solutions = []
#         fox_indices = []
        
#         # For each fox, generate candidate solutions
#         for idx, fox in enumerate(self.population):
#             s = fox['solution']
#             PF, a, m = fox['PF'], fox['a'], fox['m']
#             PF0 = self.types[fox['group'], 0]
            
#             # Generate new solution using current PF
#             flips = np.random.rand(self.num_users) < (PF / PF0)
#             if flips.sum() > 0:  # Only add if there are changes
#                 new_solution = s.copy()
#                 new_solution[flips] = np.random.randint(0, self.num_cells, flips.sum())
#                 new_solution = self.repair(new_solution)
                
#                 all_new_solutions.append(new_solution)
#                 original_solutions.append(s)
#                 fox_indices.append(idx)
                
#                 # Update PF (decay) - we'll decide whether to keep this later
#                 self.population[idx]['PF'] *= a
        
#         # If no solutions to evaluate, return early
#         if not all_new_solutions:
#             print("No solutions to evaluate in experience phase")
#             return
            
#         # Batch evaluate all new solutions
#         results = self.compute_fitness_batch(all_new_solutions)
        
#         # Process results and update foxes
#         for i, (idx, new_solution, result) in enumerate(zip(fox_indices, all_new_solutions, results)):
#             fox = self.population[idx]
#             current_fitness = fox['fitness']
#             new_fitness = result['fitness']
            
#             # If the new solution is better, update the fox
#             if new_fitness > current_fitness:
#                 fox['solution'] = new_solution
#                 fox['fitness'] = new_fitness
#                 fox['sinr'] = result.get('average_sinr', 0)
#             else:
#                 # If not improved, restore PF if it was decayed too much
#                 if fox['PF'] < m * self.types[fox['group'], 0]:
#                     fox['PF'] = m * self.types[fox['group'], 0]
                    
#         print(f"Completed batch experience phase, evaluated {len(all_new_solutions)} solutions")

#     def batch_leader_phase(self, leader_solution, max_tries=10):
#         """Batch implementation of the leader phase for all foxes"""
#         print("Starting batch leader phase...")
        
#         # Prepare batches of solutions to evaluate
#         all_new_solutions = []
#         fox_indices = []
        
#         # For each fox, generate candidate solutions based on leader
#         for idx, fox in enumerate(self.population):
#             # Skip the leader itself
#             if np.array_equal(fox['solution'], leader_solution):
#                 continue
                
#             s = fox['solution']
#             LF, b, m = fox['LF'], fox['b'], fox['m']
#             LF0 = self.types[fox['group'], 1]
            
#             # Follow leader using current LF
#             copies = np.random.rand(self.num_users) < (LF / LF0)
#             if copies.sum() > 0:  # Only add if there are changes
#                 new_solution = s.copy()
#                 new_solution[copies] = leader_solution[copies]
#                 new_solution = self.repair(new_solution)
                
#                 all_new_solutions.append(new_solution)
#                 fox_indices.append(idx)
                
#                 # Update LF (decay) - we'll decide whether to keep this later
#                 self.population[idx]['LF'] *= b
        
#         # If no solutions to evaluate, return early
#         if not all_new_solutions:
#             print("No solutions to evaluate in leader phase")
#             return
            
#         # Batch evaluate all new solutions
#         results = self.compute_fitness_batch(all_new_solutions)
        
#         # Process results and update foxes
#         for i, (idx, new_solution, result) in enumerate(zip(fox_indices, all_new_solutions, results)):
#             fox = self.population[idx]
#             current_fitness = fox['fitness']
#             new_fitness = result['fitness']
            
#             # If the new solution is better, update the fox
#             if new_fitness > current_fitness:
#                 fox['solution'] = new_solution
#                 fox['fitness'] = new_fitness
#                 fox['sinr'] = result.get('average_sinr', 0)
#             else:
#                 # If not improved, restore LF if it was decayed too much
#                 if fox['LF'] < m * self.types[fox['group'], 1]:
#                     fox['LF'] = m * self.types[fox['group'], 1]
                    
#         print(f"Completed batch leader phase, evaluated {len(all_new_solutions)} solutions")
#     def run(self, visualize_callback: callable = None, kpi_logger=None) -> np.ndarray:
#         """PFOA-aligned optimization loop with original logging structure"""
#         # Capture initial state
#         print(">>> ENTERING PFO.RUN()")
#         original_state = self.env.get_state_snapshot()
#         historical_bests = []
#         diversity_history = []
#         stagnation_threshold = 50  # PFOA's stagnation threshold
#         best_iter_metrics = None

#         # Initialize population fitness
#         for fox in self.population:            
#             fox['fitness'] = self.compute_fitness(fox['solution'])
            
#         # Initial batch evaluation
#         self.batch_evaluate_population()
        
#         best_fox = max(self.population, key=lambda f: f['fitness'])
#         self.best_solution = best_fox['solution'].copy()
#         self.best_fitness = best_fox['fitness']       
        
        
#         for iteration in range(self.iterations):
#             # 1. Experience Phase (individual foxes)
#             print(f"Entering experience phase for iteration {iteration}...")
#             print(f"\n=== Iteration {iteration+1}/{self.iterations} ===")
        
#             # 1. Batch Experience Phase (all foxes)
#             self.batch_experience_phase()
            
#             # 2. Batch Leader Phase (all foxes)
#             leader = max(self.population, key=lambda f: f['fitness'])
#             self.batch_leader_phase(leader['solution'])
            
#             # --- Update Best ---
#             # current_best = max(self.population, key=lambda f: f['fitness'])               
            
            
#             # --- Population Evaluation ---
#             fitness_values = [f['fitness'] for f in self.population]
#             current_best_idx = np.argmax(fitness_values)
#             current_best_fitness = fitness_values[current_best_idx]
#             current_best_solution = self.population[current_best_idx]['solution'].copy()
#             current_best_metrics = self.env.evaluate_detailed_solution(current_best_solution)
            
#             # --- Stagnation Tracking ---
#             if current_best_fitness > self.best_fitness * 1.001:  # 0.1% improvement threshold
#                 self.best_fitness = current_best_fitness
#                 self.best_solution = current_best_solution.copy()
#                 self.stagnation_count = 0
#             else:
#                 self.stagnation_count += 1
                
#             # --- Diversity Tracking ---
#             diversity = np.std(fitness_values)
#             diversity_history.append(diversity)
#             if len(diversity_history) > 10:  # Keep last 10 values
#                 diversity_history.pop(0)
                
#             # --- PFOA-Specific Updates ---
#             self.selective_mutation()  # Now includes PFOA's stagnation logic
#             self.update_group_weights(iteration)
#             self.fatigue_update()
            
#             # --- Logging & Visualization ---
#             historical_bests.append(self.best_fitness)
            
#             if self.kpi_logger:
#                 self.kpi_logger.log_metrics(
#                     episode=iteration,
#                     phase="metaheuristic",
#                     algorithm="pfo",
#                     metrics=current_best_metrics
#                 )
#                 print(f"Iter {iteration}: Best {self.best_fitness:.4f}, Diversity {diversity:.2f}")

#             # --- Visualization Support ---
#             if visualize_callback:
#                 viz_metrics = {
#                     "fitness": current_best_metrics["fitness"],
#                     "average_sinr": current_best_metrics["average_sinr"],
#                     "fairness": current_best_metrics["fairness"]
#                     # "diversity": diversity
#                 }
#                 visualize_callback(viz_metrics, self.best_solution)

#         # --- Finalization ---
#         self.env.set_state_snapshot(original_state)
#         self.env.apply_solution(self.best_solution)
        
#         # Prepare final metrics
#         best_iter_metrics = self.env.evaluate_detailed_solution(self.best_solution)
        
#         # # Calculate agent positions for visualization
#         # self._calculate_visual_positions()
        
#         return {
#             "solution": self.best_solution,
#             "metrics": best_iter_metrics,
#             "agents": {
#                 # "positions": self.positions.tolist(),
#                 "fitness": [f['fitness'] for f in self.population],
#                 "algorithm": "pfo"
#             }
#         }
        
import numpy as np
from envs.custom_channel_env import NetworkEnvironment 
from concurrent.futures import ThreadPoolExecutor
import copy


class PolarFoxOptimization:
    def __init__(self, env: NetworkEnvironment, iterations=20, population_size=60,
                mutation_factor=0.2, group_types: list[list[float]] | None = None, kpi_logger=None):
        self.env = env
        self.num_users = env.num_ue
        self.num_cells = env.num_bs
        self.population_size = population_size
        self.iterations = iterations
        self.mutation_factor = mutation_factor
        self.seed = 42
        np.random.seed(self.seed)
        self.kpi_logger = kpi_logger

        # PFOA CORE: Group parameters [PF0, LF0, a, b, m]
        if group_types is None:
            self.types = np.array([
                [2, 2, 0.9, 0.9, 0.1],   # Group 0
                [10, 2, 0.2, 0.9, 0.3],  # Group 1
                [2, 10, 0.9, 0.2, 0.3],  # Group 2
                [2, 12, 0.9, 0.9, 0.01]  # Group 3
            ])
        else:
            self.types = np.array(group_types, dtype=float)
        
        # PFOA CORE: Initialize population with group structures
        self.population = []
        counts = np.floor(0.25 * np.ones(4) * population_size).astype(int)
        counts[3] = population_size - counts[:3].sum()
        for k in range(4):
            for _ in range(counts[k]):
                fox = {
                    'solution': self.generate_initial_solution(),
                    'fitness': None,
                    'PF': self.types[k,0], 'LF': self.types[k,1],
                    'a': self.types[k,2], 'b': self.types[k,3], 'm': self.types[k,4],
                    'group': k
                }
                self.population.append(fox)
                
        # PFOA CORE: Weight initialization
        self.W = np.array([1000.0]*4)  # Original PFOA weights
        self.best_solution = None
        self.best_fitness = -np.inf
        self.stagnation_count = 0
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    
    def compute_fitness_batch(self, solutions):
        results = []
        def evaluate_with_env_copy(solution):
            # Create a deep copy of the environment for each evaluation
            env_copy = copy.deepcopy(self.env)
            return env_copy.evaluate_detailed_solution(solution)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(evaluate_with_env_copy, solutions))
        
        return results
    
    def vectorized_repair(self, solutions):
        """Vectorized repair for multiple solutions"""
        repaired = []
        for sol in solutions:
            counts = np.bincount(sol, minlength=self.num_cells)
            capacities = np.array([int(bs.capacity) for bs in self.env.base_stations])
            
            # Find all overloaded users at once
            overload_mask = np.isin(sol, np.where(counts > capacities)[0])
            overloaded_users = np.where(overload_mask)[0]
            
            if len(overloaded_users) > 0:
                # Batch find alternatives
                alt_bs = self._find_alternative_bs(overloaded_users, counts)
                sol[overloaded_users] = alt_bs
                
            repaired.append(sol)
        return repaired
    
    def batch_evaluate_population(self):
        """Batch evaluate all solutions with detailed results"""
        print("Starting Batch Evaluation...")
        solutions = [f['solution'] for f in self.population]
        results = self.compute_fitness_batch(solutions)
        
        for fox, result in zip(self.population, results):
            fox['fitness'] = result['fitness']
            fox['sinr'] = result.get('average_sinr', 0)  # Use get() with default in case it's missing
        print("Finished Batch evaluation..")

    # PFOA CORE: Revised experience phase with PF decay
    def experience_phase(self, fox, max_tries=10):
        s = fox['solution']
        current_fitness = self.compute_fitness(s)
        PF, a, m = fox['PF'], fox['a'], fox['m']
        PF0 = self.types[fox['group'], 0]
        
        for _ in range(max_tries):
            # Generate new solution using current PF
            flips = np.random.rand(self.num_users) < (PF / PF0)
            new_solution = s.copy()
            new_solution[flips] = np.random.randint(0, self.num_cells, flips.sum())
            new_solution = self.repair(new_solution)
            new_fitness = self.compute_fitness(new_solution)
            
            PF *= a  # Decay power factor
            if new_fitness < current_fitness or PF < m * PF0:
                break
                
        if new_fitness < current_fitness:
            fox.update({
                'solution': new_solution,
                'fitness': new_fitness,
                'PF': PF
            })

    # PFOA CORE: Revised leader phase with LF decay
    def leader_phase(self, fox, leader_solution, max_tries=10):
        s = fox['solution']
        current_fitness = fox['fitness'] or self.compute_fitness(s)
        LF, b, m = fox['LF'], fox['b'], fox['m']
        LF0 = self.types[fox['group'], 1]
        
        for _ in range(max_tries):
            # Follow leader using current LF
            copies = np.random.rand(self.num_users) < (LF / LF0)
            new_solution = s.copy()
            new_solution[copies] = leader_solution[copies]
            new_solution = self.repair(new_solution)
            new_fitness = self.compute_fitness(new_solution)
            
            LF *= b  # Decay leadership factor
            if new_fitness < current_fitness or LF < m * LF0:
                break
                
        if new_fitness < current_fitness:
            fox.update({
                'solution': new_solution,
                'fitness': new_fitness,
                'LF': LF
            })

    # PFOA CORE: Stagnation-aware mutation
    def selective_mutation(self):
        print("Mutation Phase.....")
        if self.stagnation_count > 50:  # Original PFOA condition
            num_mutate = self.population_size - 1  # Mutate all except leader
        else:
            num_mutate = int(self.mutation_factor * self.population_size)
            
        worst = sorted(self.population, key=lambda f: f['fitness'])[:num_mutate]
        for fox in worst:
            fox['solution'] = self.generate_initial_solution()
            fox['fitness'] = self.compute_fitness(fox['solution'])

    # PFOA CORE: Weight update formula W_k += I²/NG_k
    def update_group_weights(self, iteration):
        print("Updating Group weights...")
        group_counts = np.array([sum(f['group']==k for f in self.population) for k in range(4)])
        group_counts = np.clip(group_counts, 1, None)  # Avoid division by zero
        self.W += (iteration ** 2) / group_counts
        self.W /= self.W.sum()  # Normalize for probability sampling

    # PFOA CORE: Fatigue simulation from original algorithm
    def fatigue_update(self):
        print("Fatigue updates....")
        group_counts = [sum(f['group']==k for f in self.population) for k in range(4)]
        for k in range(4):
            if group_counts[k] < 5:
                # Boost parameters for rare groups
                self.types[k, 2:] = [0.99, 0.99, 0.001]
            else:
                # Decay parameters for common groups
                self.types[k, 2] = max(self.types[k,2] - 0.001, 0.9)  # a
                self.types[k, 3] = max(self.types[k,3] - 0.001, 0.9)  # b
                self.types[k, 4] = min(self.types[k,4] + 0.0001, 0.01) # m

    # *** FIXED: Network-specific methods with proper nearest cell allocation ***
    def generate_initial_solution(self):
        """Generate initial solution with 20% nearest-cell heuristic (matching first implementation)"""
        if np.random.rand() < 0.2:  # Match first implementation's 20% probability
            # *** FIXED: Use consistent UE access method ***
            # Try multiple possible attribute names to ensure compatibility
            if hasattr(self.env, 'ues'):
                user_positions = [ue.position for ue in self.env.ues]
            elif hasattr(self.env, 'users'):
                user_positions = [ue.position for ue in self.env.users]
            elif hasattr(self.env, 'user_positions'):
                user_positions = self.env.user_positions
            else:
                # Fallback: generate positions based on environment bounds
                print("Warning: Could not find UE positions, using random assignment")
                return np.random.randint(0, self.num_cells, self.num_users)
            
            # Apply nearest-cell assignment
            return np.array([self.find_nearest_cell(pos) for pos in user_positions])
        else:
            # Random assignment for remaining 80%
            return np.random.randint(0, self.num_cells, self.num_users)

    def find_nearest_cell(self, position):
        """Find nearest base station using numpy (matching first implementation)"""
        # Convert position to numpy array if needed
        pos = np.array(position, dtype=np.float32)
        
        # Get all base station positions
        cell_positions = np.stack([bs.position for bs in self.env.base_stations])
        
        # Calculate distances and return index of nearest
        distances = np.linalg.norm(cell_positions - pos, axis=1)
        return int(np.argmin(distances))
    
    def _find_alternative_bs(self, user_indices, counts):
        capacities = np.array([bs.capacity for bs in self.env.base_stations])
        available_cells = np.where(counts < capacities)[0]
        
        if len(available_cells) == 0:
            # Handle scalar/array input for user_indices
            if np.isscalar(user_indices):
                return np.random.randint(0, self.num_cells)
            else:
                return np.random.randint(0, self.num_cells, size=len(user_indices))
        
        # Find least-loaded available cell
        least_loaded = available_cells[np.argmin(counts[available_cells])]
        
        # Return same cell for all users in indices
        if np.isscalar(user_indices):
            return least_loaded
        else:
            return np.full(len(user_indices), least_loaded)

    def repair(self, solution):
        """Vectorized repair with integer casting"""
        counts = np.bincount(solution, minlength=self.num_cells)
        capacities = np.array([int(bs.capacity) for bs in self.env.base_stations])  # Ensure integer
        
        overloaded = np.where(counts > capacities)[0]
        for bs_id in overloaded:
            excess = counts[bs_id] - capacities[bs_id]
            users = np.where(solution == bs_id)[0]
            # Ensure we only move excess users using integer slicing
            for u in users[capacities[bs_id]:capacities[bs_id] + excess]:  # ✅ Integer bounds
                solution[u] = self._find_alternative_bs(u, counts)
        
        return solution
    
    def _calculate_visual_positions(self):
        """DE-style position projection (updated for PFOA structure)"""
        visual_positions = []
        for fox in self.population:  # Iterate over foxes, not raw solutions
            solution = fox["solution"]
            fitness = fox["fitness"]  # Use precomputed fitness
            
            # Feature 1: Load balance (same as before)
            counts = np.bincount(solution, minlength=self.num_cells)
            x = np.std(counts)
            
            # Feature 2: Average SINR (with temporary state safety)
            with self.env.temporary_state():
                self.env.apply_solution(solution)
                y = np.mean([ue.sinr for ue in self.env.users])
            
            visual_positions.append([x, y, fitness])  # Use stored fitness
        
        self.positions = np.array(visual_positions)
        
    def compute_fitness(self, solution):
        return self.env.evaluate_detailed_solution(solution)['fitness']

    def batch_experience_phase(self, max_tries=10):
        """Batch implementation of the experience phase for all foxes"""
        print("Starting batch experience phase...")
        
        # Prepare batches of solutions to evaluate
        all_new_solutions = []
        original_solutions = []
        fox_indices = []
        
        # For each fox, generate candidate solutions
        for idx, fox in enumerate(self.population):
            s = fox['solution']
            PF, a, m = fox['PF'], fox['a'], fox['m']
            PF0 = self.types[fox['group'], 0]
            
            # Generate new solution using current PF
            flips = np.random.rand(self.num_users) < (PF / PF0)
            if flips.sum() > 0:  # Only add if there are changes
                new_solution = s.copy()
                new_solution[flips] = np.random.randint(0, self.num_cells, flips.sum())
                new_solution = self.repair(new_solution)
                
                all_new_solutions.append(new_solution)
                original_solutions.append(s)
                fox_indices.append(idx)
                
                # Update PF (decay) - we'll decide whether to keep this later
                self.population[idx]['PF'] *= a
        
        # If no solutions to evaluate, return early
        if not all_new_solutions:
            print("No solutions to evaluate in experience phase")
            return
            
        # Batch evaluate all new solutions
        results = self.compute_fitness_batch(all_new_solutions)
        
        # Process results and update foxes
        for i, (idx, new_solution, result) in enumerate(zip(fox_indices, all_new_solutions, results)):
            fox = self.population[idx]
            current_fitness = fox['fitness']
            new_fitness = result['fitness']
            
            # If the new solution is better, update the fox
            if new_fitness > current_fitness:
                fox['solution'] = new_solution
                fox['fitness'] = new_fitness
                fox['sinr'] = result.get('average_sinr', 0)
            else:
                # If not improved, restore PF if it was decayed too much
                if fox['PF'] < m * self.types[fox['group'], 0]:
                    fox['PF'] = m * self.types[fox['group'], 0]
                    
        print(f"Completed batch experience phase, evaluated {len(all_new_solutions)} solutions")

    def batch_leader_phase(self, leader_solution, max_tries=10):
        """Batch implementation of the leader phase for all foxes"""
        print("Starting batch leader phase...")
        
        # Prepare batches of solutions to evaluate
        all_new_solutions = []
        fox_indices = []
        
        # For each fox, generate candidate solutions based on leader
        for idx, fox in enumerate(self.population):
            # Skip the leader itself
            if np.array_equal(fox['solution'], leader_solution):
                continue
                
            s = fox['solution']
            LF, b, m = fox['LF'], fox['b'], fox['m']
            LF0 = self.types[fox['group'], 1]
            
            # Follow leader using current LF
            copies = np.random.rand(self.num_users) < (LF / LF0)
            if copies.sum() > 0:  # Only add if there are changes
                new_solution = s.copy()
                new_solution[copies] = leader_solution[copies]
                new_solution = self.repair(new_solution)
                
                all_new_solutions.append(new_solution)
                fox_indices.append(idx)
                
                # Update LF (decay) - we'll decide whether to keep this later
                self.population[idx]['LF'] *= b
        
        # If no solutions to evaluate, return early
        if not all_new_solutions:
            print("No solutions to evaluate in leader phase")
            return
            
        # Batch evaluate all new solutions
        results = self.compute_fitness_batch(all_new_solutions)
        
        # Process results and update foxes
        for i, (idx, new_solution, result) in enumerate(zip(fox_indices, all_new_solutions, results)):
            fox = self.population[idx]
            current_fitness = fox['fitness']
            new_fitness = result['fitness']
            
            # If the new solution is better, update the fox
            if new_fitness > current_fitness:
                fox['solution'] = new_solution
                fox['fitness'] = new_fitness
                fox['sinr'] = result.get('average_sinr', 0)
            else:
                # If not improved, restore LF if it was decayed too much
                if fox['LF'] < m * self.types[fox['group'], 1]:
                    fox['LF'] = m * self.types[fox['group'], 1]
                    
        print(f"Completed batch leader phase, evaluated {len(all_new_solutions)} solutions")

    def run(self, visualize_callback: callable = None, kpi_logger=None) -> np.ndarray:
        """PFOA-aligned optimization loop with original logging structure"""
        # Capture initial state
        print(">>> ENTERING PFO.RUN()")
        original_state = self.env.get_state_snapshot()
        historical_bests = []
        diversity_history = []
        stagnation_threshold = 50  # PFOA's stagnation threshold
        best_iter_metrics = None

        # Initialize population fitness
        for fox in self.population:            
            fox['fitness'] = self.compute_fitness(fox['solution'])
            
        # Initial batch evaluation
        self.batch_evaluate_population()
        
        best_fox = max(self.population, key=lambda f: f['fitness'])
        self.best_solution = best_fox['solution'].copy()
        self.best_fitness = best_fox['fitness']       
        
        
        for iteration in range(self.iterations):
            # 1. Experience Phase (individual foxes)
            print(f"Entering experience phase for iteration {iteration}...")
            print(f"\n=== Iteration {iteration+1}/{self.iterations} ===")
        
            # 1. Batch Experience Phase (all foxes)
            self.batch_experience_phase()
            
            # 2. Batch Leader Phase (all foxes)
            leader = max(self.population, key=lambda f: f['fitness'])
            self.batch_leader_phase(leader['solution'])
            
            # --- Update Best ---
            # current_best = max(self.population, key=lambda f: f['fitness'])               
            
            
            # --- Population Evaluation ---
            fitness_values = [f['fitness'] for f in self.population]
            current_best_idx = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]
            current_best_solution = self.population[current_best_idx]['solution'].copy()
            current_best_metrics = self.env.evaluate_detailed_solution(current_best_solution)
            
            # --- Stagnation Tracking ---
            if current_best_fitness > self.best_fitness * 1.001:  # 0.1% improvement threshold
                self.best_fitness = current_best_fitness
                self.best_solution = current_best_solution.copy()
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1
                
            # --- Diversity Tracking ---
            diversity = np.std(fitness_values)
            diversity_history.append(diversity)
            if len(diversity_history) > 10:  # Keep last 10 values
                diversity_history.pop(0)
                
            # --- PFOA-Specific Updates ---
            self.selective_mutation()  # Now includes PFOA's stagnation logic
            self.update_group_weights(iteration)
            self.fatigue_update()
            
            # --- Logging & Visualization ---
            historical_bests.append(self.best_fitness)
            
            if self.kpi_logger:
                self.kpi_logger.log_metrics(
                    episode=iteration,
                    phase="metaheuristic",
                    algorithm="pfo",
                    metrics=current_best_metrics
                )
                print(f"Iter {iteration}: Best {self.best_fitness:.4f}, Diversity {diversity:.2f}")

            # --- Visualization Support ---
            if visualize_callback:
                viz_metrics = {
                    "fitness": current_best_metrics["fitness"],
                    "average_sinr": current_best_metrics["average_sinr"],
                    "fairness": current_best_metrics["fairness"]
                    # "diversity": diversity
                }
                visualize_callback(viz_metrics, self.best_solution)

        # --- Finalization ---
        self.env.set_state_snapshot(original_state)
        self.env.apply_solution(self.best_solution)
        
        # Prepare final metrics
        best_iter_metrics = self.env.evaluate_detailed_solution(self.best_solution)
        
        # # Calculate agent positions for visualization
        # self._calculate_visual_positions()
        
        return {
            "solution": self.best_solution,
            "metrics": best_iter_metrics,
            "agents": {
                # "positions": self.positions.tolist(),
                "fitness": [f['fitness'] for f in self.population],
                "algorithm": "pfo"
            }
        }      
        
        
# # # algorithms/pfo.py

# import numpy as np
# import torch
# from envs.custom_channel_env import NetworkEnvironment

# class PolarFoxOptimization:
#     def __init__(self, env: NetworkEnvironment, iterations=20,population_size = 30, mutation_factor = 0.2,
#                 jump_rate = 0.2,follow_rate = 0.3,group_weights: list[float] | None = None,kpi_logger=None):
                
#         self.env = env
#         self.num_users = env.num_ue
#         self.num_cells = env.num_bs        
#         self.population_size = population_size
#         self.iterations = iterations
#         self.mutation_factor = mutation_factor
#         self.jump_rate = jump_rate 
#         self.follow_rate = follow_rate  
#         self.seed = 42
#         np.random.seed(self.seed)
#         self.kpi_logger = kpi_logger
        
#         # Group parameters [PF, LF, a, b, m]
#         self.types = np.array([
#             [2, 2, 0.9, 0.9, 0.1],   # Group 0: Balanced
#             [10, 2, 0.2, 0.9, 0.3],  # Group 1: Explorer
#             [2, 10, 0.9, 0.2, 0.3],  # Group 2: Follower
#             [2, 12, 0.9, 0.9, 0.01]  # Group 3: Conservative
#         ])
        
#         # Initialize population with groups
#         self.population = self.initialize_population(env)
#         # self.group_weights = [1000, 1000, 1000, 1000]
#          # Initialize group weights: use provided or default to equal
#         if group_weights is not None:
#             raw = np.array(group_weights, dtype=float)
#         else:
#             raw = np.ones(4, dtype=float)
#         # Normalize so they sum to 1 for probability sampling
#         self.group_weights = (raw / raw.sum()).tolist()
#         self.best_solution = None
#         self.best_fitness = -np.inf
#         # For live updates, you can also keep track of positions and fitness history:
#         # self.positions = None  # will be computed from population later        
#         self.positions = np.empty((0, 3))  # Match DE's 3D position format
#         self.fitness = np.full(self.iterations, np.nan)  # Pre-allocate fitness array
#         self.best_fitness_history = []  # Rename from historical_bests
#         self.best_metrics_history = []  # To store metrics per iteration
#         self._rng = np.random.RandomState(self.seed)  # DE-style RNG
        
#     def initialize_population(self, env: NetworkEnvironment):
#         population = []
#         for _ in range(self.population_size):
#             if np.random.rand() < 0.2:
#                 # Heuristic: Assign users to nearest cell
#                 fox = np.array([self.find_nearest_cell(ue.position) for ue in env.ues])
#             else:
#                 fox = np.random.randint(0, self.num_cells, size=self.num_users)
#             population.append(fox)
#         # print(f"Population initialized: {population}")
#         return population


#     # def find_nearest_cell(self, user_position):
#     #     if not isinstance(user_position, torch.Tensor):
#     #         user_position = torch.tensor(user_position, dtype=torch.float32)
#     #     cell_positions = torch.stack([bs.position for bs in self.env.base_stations]).to(user_position.device)
#     #     with torch.no_grad():
#     #         distances = torch.norm(cell_positions - user_position, dim=1)

#     #     # distances = torch.norm(cell_positions - user_position, dim=1)
#     #     return torch.argmin(distances).item()   

#     def find_nearest_cell(self, user_position):
#         # use numpy instead of torch
#         pos = np.array(user_position, dtype=np.float32)
#         cell_positions = np.stack([bs.position for bs in self.env.base_stations])
#         dists = np.linalg.norm(cell_positions - pos, axis=1)
#         return int(np.argmin(dists))

    
#     def _find_alternative_bs(self, user_indices, counts):
#         capacities = np.array([bs.capacity for bs in self.env.base_stations])
#         available_cells = np.where(counts < capacities)[0]
#         if len(available_cells) == 0:
#             return np.random.randint(0, self.num_cells, size=len(user_indices))
#         new_assignment = available_cells[np.argmin(counts[available_cells])]
#         return new_assignment
    
#     def calculate_group_distribution(self):
#         """Distribute foxes according to initial group ratios"""
#         base_dist = np.array([0.25, 0.25, 0.25, 0.25])
#         counts = np.floor(base_dist * self.population_size).astype(int)
#         counts[3] = self.population_size - sum(counts[:3])
#         return counts

#     def create_fox(self, group_id):
#         """Create fox with group-specific parameters"""
#         fox = {
#             "solution": self.generate_initial_solution(),
#             "PF": self.types[group_id, 0],
#             "LF": self.types[group_id, 1],
#             "a": self.types[group_id, 2],
#             "b": self.types[group_id, 3],
#             "m": self.types[group_id, 4],
#             "group": group_id
#         }
#         return fox

#     # algorithms/pfo.py (modified)
#     def generate_initial_solution(self):
#         """20% use nearest-cell heuristic"""
#         if np.random.rand() < 0.2:
#             # Replace self.env.users with self.env.user_positions
#             return np.array([self.env.find_nearest_cell(pos) for pos in self.env.user_positions])
#         else:
#             return np.random.randint(0, self.num_cells, size=self.num_users)

#     def compute_fitness(self, solution):
#         return self.env.evaluate_detailed_solution(solution)["fitness"]

#     def adaptive_parameters(self, iteration):
#         """Decay jump power (a), boost follow power (b) over time"""
#         decay_factor = 1 - (iteration / self.iterations) ** 0.5
#         for i in range(4):
#             self.types[i, 2] = max(self.types[i, 2] * decay_factor, 0.1)
#             self.types[i, 3] = min(self.types[i, 3] / decay_factor, 0.99)

#     def jump_experience(self, fox):
#         """Randomly jump to explore new solutions."""
#         new_fox = fox.copy()
#         num_jumps = int(self.num_users * self.jump_rate)  # Convert to integer
#         if num_jumps < 1:  # Ensure at least 1 jump
#             num_jumps = 1
#         indices = np.random.choice(self.num_users, num_jumps, replace=False)
#         new_fox[indices] = np.random.randint(0, self.num_cells, size=num_jumps)
#         return new_fox

#     def follow_leader(self, current_fox, best_fox):
#         """Update part of the solution to follow the best solution."""
#         new_fox = current_fox.copy()
#         num_follow = int(self.num_users * self.follow_rate)  # <-- FIXED
#         indices = np.random.choice(self.num_users, num_follow, replace=False)
#         new_fox[indices] = best_fox[indices]
#         return new_fox

#     # def repair(self, solution):
#     #     """Ensure cell capacity constraints"""
#     #     cell_counts = np.bincount(solution, minlength=self.num_cells)
#     #     overloaded = np.where(cell_counts > self.env.cell_capacity)[0]
        
#     #     for cell in overloaded:
#     #         users = np.where(solution == cell)[0]
#     #         for user in users[self.env.cell_capacity:]:
#     #             solution[user] = np.argmin(cell_counts)
#     #             cell_counts[solution[user]] += 1
#     #     return solution
    
#     def repair(self, solution):
#         """DE-style vectorized repair"""
#         counts = np.bincount(solution, minlength=self.num_cells)
#         capacities = np.array([bs.capacity for bs in self.env.base_stations])
        
#         overloaded = np.where(counts > capacities)[0]
#         for bs_id in overloaded:
#             excess = counts[bs_id] - capacities[bs_id]
#             users = np.where(solution == bs_id)[0][:excess]
#             solution[users] = self._find_alternative_bs(users, counts)
        
#         return solution
    
#     # def leader_motivation(self, stagnation_count):
#     #     """Reset underperforming foxes and adjust groups"""
#     #     num_mutation = int(self.population_size * self.mutation_factor)
#     #     for i in range(num_mutation):
#     #         group_id = np.random.choice(4, p=self.group_weights/np.sum(self.group_weights))
#     #         self.population[i] = self.create_fox(group_id)
        
#     #     # Boost weights of best-performing group
#     #     if self.best_solution is not None:
#     #         best_group = self.population[np.argmax([f["group"] for f in self.population])]["group"]
#     #         self.group_weights[best_group] += stagnation_count * 100
#     def leader_motivation(self, stagnation_count: int):
#         """Reset underperforming foxes and adjust group selection probabilities."""
#         num_mutation = int(self.population_size * self.mutation_factor)
#         for i in range(num_mutation):
#             # Sample group according to normalized weights
#             group_id = np.random.choice(4, p=self.group_weights)
#             self.population[i] = self.create_fox(group_id)

#         # Boost weights of best-performing group, then renormalize
#         if self.best_solution is not None:
#             best_group = self.population[
#                 np.argmax([f['group'] for f in self.population])
#             ]['group']
#             # Increase weight for best group
#             gw = np.array(self.group_weights, dtype=float)
#             gw[best_group] += stagnation_count * 100
#             # Renormalize probabilities
#             self.group_weights = (gw / gw.sum()).tolist()

            
#     def _calculate_visual_positions(self):
#         """DE-style position projection"""
#         visual_positions = []
#         for solution in self.population:
#             # Feature 1: Load balance
#             counts = np.bincount(solution, minlength=self.num_cells)
#             x = np.std(counts)
            
#             # Feature 2: Average SINR (via temporary env state)
#             with self.env.temporary_state():
#                 self.env.apply_solution(solution)
#                 y = np.mean([ue.sinr for ue in self.env.users])
            
#             visual_positions.append([x, y, self.fitness(solution)])
#         self.positions = np.array(visual_positions)    

    
#     def run(self,visualize_callback: callable = None, kpi_logger=None) -> np.ndarray:
#         """Enhanced optimization loop with anti-stagnation mechanisms."""
#         # 🔴 Capture initial state
#         original_state = self.env.get_state_snapshot()
#         best_solution = self.population[0].copy()
#         best_fitness = -np.inf
#         historical_bests = []
#         no_improvement_streak = 0
#         stagnation_threshold = 5  # Increased from 3
#         diversity_window = 10  # Track diversity over last N iterations
#         mutation_reset = 0.1  # Minimum mutation factor
        
#         # Initialize population diversity tracking
#         diversity_history = []
#         # current_metrics = env.evaluate_detailed_solution(current_solution)
#         best_iter_metrics = self.env.evaluate_detailed_solution(best_solution)
        
#         for iteration in range(self.iterations):
#             # Adaptive parameter update
#             self.adaptive_parameters(iteration)
            
#             # Evaluate population
#             fitness_values = [self.compute_fitness(fox) for fox in self.population]
            
#             # Update best solution with elitism
#             current_best_idx = np.argmax(fitness_values)
#             current_best_fitness = fitness_values[current_best_idx]
#             current_best_solution = self.population[current_best_idx].copy()
#             current_best_metrics = self.env.evaluate_detailed_solution(current_best_solution)

            
#             # Maintain diversity tracking
#             diversity = np.std(fitness_values)
#             diversity_history.append(diversity)
#             if len(diversity_history) > diversity_window:
#                 diversity_history.pop(0)

#             # Update best solution with momentum (prevent oscillation)
#             if current_best_fitness > best_fitness * 1.001:  # 0.1% improvement threshold
#                 best_fitness = current_best_fitness
#                 best_solution = self.population[current_best_idx].copy()
#                 no_improvement_streak = 0
#             else:
#                 no_improvement_streak += 1

#             # Log metrics per iteration via KPI logger if available
#             if self.kpi_logger:
#                 self.kpi_logger.log_metrics(
#                     episode=iteration,
#                     phase="metaheuristic",
#                     algorithm="pfo",
#                     metrics=current_best_metrics # Log full metrics {"fitness": best_fitness, "diversity": diversity}
#                 )
#                 print(f"PFO Iter {iteration}: Best Fitness = {best_fitness:.4f}, Diversity = {diversity:.2f}")
#             print(f"Logging metrics at episode {iteration}: {current_best_metrics}")
#             historical_bests.append(best_fitness)
#             # prepare the simple metrics dict for plotting
             
#             # # Periodic live dashboard updates (every 5 iterations)
#             # if visualize_callback and iteration % 5 == 0:
#             #     # Compute visualization positions (example: use diversity as x and best fitness as y)
#             #     positions = np.column_stack((
#             #         np.full((self.population_size,), diversity),   # Dummy x: diversity for all
#             #         np.array(fitness_values)                        # y: fitness values
#             #     ))
#             #     self.positions = positions  # Save current positions for visualization
                
#             #     visualize_callback({
#             #         "positions": positions.tolist(),
#             #         "fitness": self.best_fitness_history,
#             #         "algorithm": "pfo",
#             #         "env_state": self.env.get_current_state()
#             #     })
#             #     print(f"PFO Visual Update @ Iter {iteration}")
                
#             # Update global best solution
#             if current_best_metrics["fitness"] > self.best_fitness:
#                 self.best_fitness = current_best_metrics["fitness"]
#                 best_solution = current_best_solution.copy()
                
#             # Enhanced stagnation detection with diversity check
#             avg_diversity = np.mean(diversity_history[-3:]) if diversity_history else 0
#             if (iteration > 20 and 
#                 no_improvement_streak >= stagnation_threshold and 
#                 avg_diversity < 0.5 * np.mean(diversity_history)):
                
#                 # Aggressive mutation boost
#                 self.mutation_factor = min(1.0, self.mutation_factor * 2)
#                 no_improvement_streak = max(0, no_improvement_streak - 2)
                
#                 # Diversity injection
#                 num_replace = int(0.2 * self.population_size)
#                 for i in range(num_replace):
#                     self.population[-(i+1)] = self.random_solution()
                
#                 print(f"Iter {iteration}: Mutation ↑ {self.mutation_factor:.2f}, Diversity injection")

#             # Dynamic population management
#             sorted_indices = np.argsort(fitness_values)[::-1]
            
#             # Keep top 10% elites unchanged
#             elite_count = max(1, int(0.1 * self.population_size))
#             elites = [self.population[i].copy() for i in sorted_indices[:elite_count]]
            
#             # Generate new population
#             new_population = elites.copy()
            
#             # Create remaining population through enhanced operations
#             while len(new_population) < self.population_size:
#                 parent = self.population[np.random.choice(sorted_indices[:elite_count*2])]
                
#                 if np.random.rand() < 0.7:  # Favor exploitation
#                     child = self.follow_leader(parent, best_solution)
#                 else:  # Exploration
#                     child = self.jump_experience(parent)
                    
#                 # Apply mutation with adaptive probability
#                 mutation_prob = 0.3 + (0.5 * (self.mutation_factor / 1.0))
#                 if np.random.rand() < mutation_prob:
#                     child = self.jump_experience(child)
                    
#                 new_population.append(child)

#             self.population = new_population
            
#             # Adaptive mutation decay
#             if no_improvement_streak == 0 and self.mutation_factor > mutation_reset:
#                 self.mutation_factor *= 0.95  # Gradual decay on improvement
                
#             self.best_solution = best_solution.copy()    
#             # Update environment with current best solution
#             viz_metrics = {
#                 "fitness":        current_best_metrics["fitness"],
#                 "average_sinr":   current_best_metrics["average_sinr"],
#                 "fairness":       current_best_metrics["fairness"]
#             }

#             # call the Streamlit callback (if any), passing metrics and the best solution
#             if visualize_callback:
#                 visualize_callback(viz_metrics, self.best_solution)
#         # 🔴 Restore environment after optimization
#         self.env.set_state_snapshot(original_state)
        
#         self.env.apply_solution(self.best_solution)
#         # actions = {
#         #         f"bs_{bs_id}": np.where(self.best_solution == bs_id)[0].tolist()
#         #         for bs_id in range(self.env.num_bs)
#         #     }
#         # self.env.step(actions)  # Update environment state
            
#             # # Progress tracking
#             # historical_bests.append(best_fitness)
#             # print(f"Iter {iteration+1}: Best = {best_fitness:.4f}, "
#             #     f"Mutation = {self.mutation_factor:.2f}, "
#             #     f"Diversity = {diversity:.2f}")

#         # return best_solution
#         # Return DE-style output
#         return {
#             "solution": self.best_solution,
#             "metrics": best_iter_metrics,
#             "agents": {
#                 "positions": self.positions.tolist(),
#                 "fitness": self.fitness.tolist(),
#                 "algorithm": "pfo"
#             }
#         }
    