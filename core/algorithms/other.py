
# algorithms/pfo.py
import numpy as np
import torch
import types
import time
from concurrent.futures import ThreadPoolExecutor
from envs.custom_channel_env import NetworkEnvironment

# Monkey-patch: batched evaluation for speed
# Monkey-patch: batched evaluation for speed
def _batch_evaluate(self, solutions: torch.LongTensor) -> torch.Tensor:
    """
    Batch-evaluate B assignments in parallel using tensor ops.
    solutions: (B, U) long tensor of UE->BS assignments.
    Returns: (B,) tensor of fitness values.
    """
    B, U = solutions.shape
    device = solutions.device
    # replicate UE & BS positions
    ue_pos = self.ues.position.unsqueeze(0).expand(B, U, -1)
    bs_pos = self.base_stations.position.unsqueeze(0).unsqueeze(1).expand(B, U, -1, -1)
    # gather chosen BS coords
    picks = solutions.unsqueeze(-1).unsqueeze(-1).expand(B, U, 1, self.bs_positions.size(-1))
    ue_bs_pos = torch.gather(bs_pos, 2, picks).squeeze(2)  # (B, U, dim)
    # compute SINR batch
    sinr = self._batch_compute_sinr(ue_bs_pos)  # implement in env for batch
    avg_sinr = sinr.mean(dim=1)                 # (B,)
    # compute load-balance penalty
    counts = torch.zeros((B, self.num_bs), device=device)
    counts.scatter_add_(1, solutions, torch.ones_like(solutions, dtype=torch.float, device=device))
    load_std = counts.std(dim=1)                # (B,)
    # fitness: higher is better
    return avg_sinr - load_std

# Attach to environment
NetworkEnvironment.batch_evaluate = _batch_evaluate
# Monkey-patch apply_solution to handle list vs dict for self.ues
# Capture original apply_solution
_orig_apply_solution = NetworkEnvironment.apply_solution

def _apply_solution_safe(self, solution):
    # if ues is list, convert to dict for PRB allocation
    list_flag = isinstance(self.ues, list)
    if list_flag:
        ues_list = self.ues
        self.ues = {ue.id: ue for ue in ues_list}
    try:
        return _orig_apply_solution(self, solution)
    finally:
        if list_flag:
            self.ues = ues_list

# Override the method
NetworkEnvironment.apply_solution = _apply_solution_safe



class PolarFoxOptimization:
    def __init__(self,
                 env: NetworkEnvironment,
                 iterations: int = 20,
                 population_size: int = 30,
                 mutation_factor: float = 0.2,
                 max_tries: int = 8,
                 stagnation_threshold: int = 50,
                 group_weights: list[float] = None,
                 kpi_logger=None):
        self.env = env
        # bind batch evaluate
        self.env.batch_evaluate = types.MethodType(_batch_evaluate, self.env)
        # cache environment data
        self.ue_positions = torch.tensor([ue.position for ue in env.ues], dtype=torch.float32)  # (U, dim) list -> tensor
        self.bs_positions = torch.tensor([bs.position for bs in env.base_stations], dtype=torch.float32)  # (num_bs, dim) list -> tensor
        self.device = self.ue_positions.device
        self.num_users = env.num_ue
        self.num_cells = env.num_bs
        # parameters
        self.iterations = iterations
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.max_tries = max_tries
        self.stagnation_threshold = stagnation_threshold
        self.kpi_logger = kpi_logger
        # PFO types: [PF0, LF0, a0, b0, m0]
        self.types = np.array([
            [2, 2, 0.9, 0.9, 0.1],
            [10,2, 0.2, 0.9, 0.3],
            [2,10, 0.9, 0.2, 0.3],
            [2,12, 0.9, 0.9, 0.01]
        ])
        raw = np.ones(4) if group_weights is None else np.array(group_weights, float)
        self.W = raw / raw.sum()
        # precompute nearest-cell for heuristic init
        bs_np = np.stack([bs.position for bs in env.base_stations])  # (num_bs, dim)  # (U,dim)
        ue_np = np.stack([ue.position for ue in env.ues])  # (num_ue, dim)            # (U,dim)
        dists = np.linalg.norm(ue_np[:,None,:] - bs_np[None,:,:], axis=2)
        self._nearest = np.argmin(dists, axis=1)
        # initialize population
        self.population = []
        counts = np.floor(0.25 * np.ones(4) * population_size).astype(int)
        counts[3] = population_size - counts[:3].sum()
        for k, cnt in enumerate(counts):
            for _ in range(cnt):
                sol = self._initial_solution()
                self.population.append({
                    'sol': sol,
                    'fitness': None,
                    'PF0': self.types[k,0], 'LF0': self.types[k,1],
                    'a0': self.types[k,2], 'b0': self.types[k,3], 'm0': self.types[k,4],
                    'PF': self.types[k,0], 'LF': self.types[k,1],
                    'type': k
                })
    def _initial_solution(self) -> np.ndarray:
        if np.random.rand() < 0.2:
            return self._nearest.copy()
        else:
            return np.random.randint(0, self.num_cells, size=self.num_users)
    def compute_fitness(self, sol: np.ndarray) -> float:
        """
        Compute scalar fitness via batched tensor evaluation, bypassing detailed PRB allocation.
        """
        # convert to torch tensor batch shape (1, U)
        t = torch.tensor(sol, device=self.device, dtype=torch.long).unsqueeze(0)
        # batch_evaluate returns (B,) where B=1
        fit = self.env.batch_evaluate(t)[0]
        return float(fit)
    def experience_phase(self, fox: dict):
        s = torch.tensor(fox['sol'], device=self.device).long()  # (U,)
        c = fox['fitness']
        PF, PF0, a0, m0 = fox['PF'], fox['PF0'], fox['a0'], fox['m0']
        T = min(self.max_tries, int(np.ceil(PF / PF0 * self.max_tries)))
        # generate T candidates
        masks = torch.rand((T, self.num_users), device=self.device) < (PF / PF0)
        base = s.unsqueeze(0).expand(T, -1)
        rand_asg = torch.randint(0, self.num_cells, (T, self.num_users), device=self.device)
        cand = base.clone()
        cand[masks] = rand_asg[masks]
        fitnesses = self.env.batch_evaluate(cand)
        # pick first improving
        better = (fitnesses < c).nonzero(as_tuple=True)[0]
        if better.numel() > 0:
            j = better[0].item()
            new = cand[j].cpu().numpy()
            fox.update({'sol': new, 'fitness': fitnesses[j].item(), 'PF': PF * (a0 ** (j+1))})
        else:
            fox['PF'] = PF * (a0 ** T)
    def leader_phase(self, fox: dict, best: np.ndarray):
        s = torch.tensor(fox['sol'], device=self.device).long()
        c = fox['fitness']
        LF, LF0, b0, m0 = fox['LF'], fox['LF0'], fox['b0'], fox['m0']
        T = min(self.max_tries, int(np.ceil(LF / LF0 * self.max_tries)))
        masks = torch.rand((T, self.num_users), device=self.device) < (LF / LF0)
        base = s.unsqueeze(0).expand(T, -1)
        leader = torch.tensor(best, device=self.device).long().unsqueeze(0).expand(T, -1)
        cand = base.clone()
        cand[masks] = leader[masks]
        fitnesses = self.env.batch_evaluate(cand)
        better = (fitnesses < c).nonzero(as_tuple=True)[0]
        if better.numel() > 0:
            j = better[0].item()
            new = cand[j].cpu().numpy()
            fox.update({'sol': new, 'fitness': fitnesses[j].item(), 'LF': LF * (b0 ** (j+1))})
        else:
            fox['LF'] = LF * (b0 ** T)
    def selective_mutation(self):
        nm = int(self.mutation_factor * self.population_size)
        # assume population sorted descending
        worst = self.population[-nm:]
        for fox in worst:
            sol = self._initial_solution()
            fox.update({'sol': sol, 'fitness': self.compute_fitness(sol), 'PF': fox['PF0'], 'LF': fox['LF0']})
    def update_weights_and_motivate(self, best_type: int, iteration: int):
        counts = [sum(f['type'] == k for f in self.population) for k in range(4)]
        # update W
        self.W[best_type] += iteration**2 / max(1, counts[best_type])
        probs = self.W / self.W.sum()
        # reassign types for non-leader
        for fox in self.population[1:]:
            k = np.random.choice(4, p=probs)
            fox.update({
                'type': k,
                'PF0': self.types[k,0], 'LF0': self.types[k,1],
                'a0': self.types[k,2], 'b0': self.types[k,3], 'm0': self.types[k,4],
                'PF': self.types[k,0], 'LF': self.types[k,1]
            })
    def fatigue_update(self):
        counts = [sum(f['type'] == k for f in self.population) for k in range(4)]
        for k in range(4):
            if counts[k] < 5:
                self.types[k,2:] = [0.99, 0.99, 0.001]
            else:
                self.types[k,2] = max(self.types[k,2] - 0.001, 0.9)
                self.types[k,3] = max(self.types[k,3] - 0.001, 0.9)
                self.types[k,4] = min(self.types[k,4] + 0.0001, 0.01)
    # def run(self, visualize_callback=None):
    def run(self,visualize_callback: callable = None, kpi_logger=None) -> np.ndarray:
        # snapshot environment state
        original_state = self.env.get_state_snapshot()
        # Ensure env.ues is a dict (allocate_prbs expects dict[ue_id])
        ues_original = self.env.ues
        if isinstance(self.env.ues, list):
            self.env.ues = {ue.id: ue for ue in self.env.ues}
        # initial fitness via threads
        sols = [f['sol'] for f in self.population]
        with ThreadPoolExecutor() as ex:
            fits = list(ex.map(self.compute_fitness, sols))
        for f, fit in zip(self.population, fits):
            f['fitness'] = fit
        # sort descending
        self.population.sort(key=lambda f: f['fitness'], reverse=True)
        self.best = self.population[0]['sol'].copy()
        self.best_fit = self.population[0]['fitness']
        stagnation = 0

        for I in range(self.iterations):
            start = time.time()
            # Phases
            for fox in self.population:
                self.experience_phase(fox)
            for fox in self.population:
                self.leader_phase(fox, self.best)
            print(f"Iteration {I+1} core phases: {time.time()-start:.2f}s")

            # sort & check
            self.population.sort(key=lambda f: f['fitness'], reverse=True)
            if self.population[0]['fitness'] <= self.best_fit + 1e-8:
                stagnation += 1
            else:
                self.best_fit = self.population[0]['fitness']
                self.best = self.population[0]['sol'].copy()
                stagnation = 0
            # stagnation reset
            if stagnation > self.stagnation_threshold:
                # reset all but leader
                for fox in self.population[1:]:
                    sol = self._initial_solution()
                    fox.update({'sol': sol, 'fitness': self.compute_fitness(sol), 'PF': fox['PF0'], 'LF': fox['LF0']})
                stagnation = 0

            # mutation, motivation, fatigue
            self.selective_mutation()
            self.update_weights_and_motivate(self.population[0]['type'], I)
            self.fatigue_update()

            # logging & visualization
            if self.kpi_logger:
                self.kpi_logger.log_metrics(episode=I,  phase="metaheuristic",
                                            algorithm='pfo', metrics={'fitness': self.best_fit})
            print(f"End Iter {I+1}: Best fitness={self.best_fit:.4f}")
            if visualize_callback:
                visualize_callback({'best_fit': self.best_fit, 'population': [f['fitness'] for f in self.population]})

        # restore environment state
        self.env.set_state_snapshot(original_state)
        # finalize by applying best solution
        self.env.apply_solution(self.best)
        return {'solution': self.best, 'best_fitness': self.best_fit}

           

 
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
import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class PolarFoxOptimization:
    def __init__(self, env: NetworkEnvironment, iterations=20, population_size=30,
                 mutation_factor=0.2, group_weights: list[float] | None = None, kpi_logger=None):
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
        self.types = np.array([
            [2, 2, 0.9, 0.9, 0.1],   # Group 0
            [10, 2, 0.2, 0.9, 0.3],  # Group 1
            [2, 10, 0.9, 0.2, 0.3],  # Group 2
            [2, 12, 0.9, 0.9, 0.01]  # Group 3
        ])
        
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

    # PFOA CORE: Revised experience phase with PF decay
    def experience_phase(self, fox):
        s = fox['solution']
        current_fitness = self.compute_fitness(s)
        PF, a, m = fox['PF'], fox['a'], fox['m']
        PF0 = self.types[fox['group'], 0]
        
        while True:
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
    def leader_phase(self, fox, leader_solution):
        s = fox['solution']
        current_fitness = fox['fitness'] or self.compute_fitness(s)
        LF, b, m = fox['LF'], fox['b'], fox['m']
        LF0 = self.types[fox['group'], 1]
        
        while True:
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
        group_counts = np.array([sum(f['group']==k for f in self.population) for k in range(4)])
        group_counts = np.clip(group_counts, 1, None)  # Avoid division by zero
        self.W += (iteration ** 2) / group_counts
        self.W /= self.W.sum()  # Normalize for probability sampling

    # PFOA CORE: Fatigue simulation from original algorithm
    def fatigue_update(self):
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

    # Maintained network-specific methods
    def generate_initial_solution(self):
        if np.random.rand() < 0.2:
            return np.array([self.find_nearest_cell(ue.position) for ue in self.env.ues])
        return np.random.randint(0, self.num_cells, self.num_users)

    def find_nearest_cell(self, position):
        cell_positions = np.stack([bs.position for bs in self.env.base_stations])
        return np.argmin(np.linalg.norm(cell_positions - position, axis=1))
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

    def run(self, visualize_callback: callable = None, kpi_logger=None) -> np.ndarray:
        """PFOA-aligned optimization loop with original logging structure"""
        # Capture initial state
        original_state = self.env.get_state_snapshot()
        historical_bests = []
        diversity_history = []
        stagnation_threshold = 50  # PFOA's stagnation threshold
        best_iter_metrics = None

        # Initialize population fitness
        for fox in self.population:
            fox['fitness'] = self.compute_fitness(fox['solution'])
        
        best_fox = max(self.population, key=lambda f: f['fitness'])
        self.best_solution = best_fox['solution'].copy()
        self.best_fitness = best_fox['fitness']
        
        for iteration in range(self.iterations):
            # 1. Individual experience phase (with repair)
            for fox in self.population:
                self.experience_phase(fox)  # Has internal repair
                
            # 2. Individual leader phase (with repair)
            leader = max(self.population, key=lambda f: f['fitness'])
            for fox in self.population:
                self.leader_phase(fox, leader['solution'])  # Internal repair
                
            # 3. Batch evaluate (parallel)
            self.batch_evaluate_population()
            
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
                    metrics={
                        **current_best_metrics,
                        'diversity': diversity,
                        'stagnation': self.stagnation_count
                    }
                )
                print(f"Iter {iteration}: Best {self.best_fitness:.4f}, Diversity {diversity:.2f}")

            # --- Visualization Support ---
            if visualize_callback:
                viz_metrics = {
                    "fitness": current_best_metrics["fitness"],
                    "average_sinr": current_best_metrics["average_sinr"],
                    "fairness": current_best_metrics["fairness"],
                    "diversity": diversity
                }
                visualize_callback(viz_metrics, self.best_solution)

        # --- Finalization ---
        self.env.set_state_snapshot(original_state)
        self.env.apply_solution(self.best_solution)
        
        # Prepare final metrics
        best_iter_metrics = self.env.evaluate_detailed_solution(self.best_solution)
        
        # Calculate agent positions for visualization
        self._calculate_visual_positions()
        
        return {
            "solution": self.best_solution,
            "metrics": best_iter_metrics,
            "agents": {
                "positions": self.positions.tolist(),
                "fitness": [f['fitness'] for f in self.population],
                "algorithm": "pfo"
            }
        }