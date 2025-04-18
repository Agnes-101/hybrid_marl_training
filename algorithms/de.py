# algorithms/de.py
import numpy as np
from collections import deque
from envs.custom_channel_env import NetworkEnvironment

class DEOptimization:
    def __init__(self, env, kpi_logger=None):
        """Differential Evolution for 6G UE-BS Association"""
        # Optimization parameters
        self.env=env
        self.population_size = 30
        self.iterations = 20
        self.F = 1.2  # Mutation factor
        self.CR = 0.5  # Crossover rate
        
        # State tracking
        
        self.positions = np.empty((0, 3))  # Proper initialization
        # self.fitness = []  # Track fitness per candidate solution
        self.fitness = np.full(self.iterations, np.nan)  # Pre-allocate with NaNs
        self.best_fitness_history = []  # Track best fitness per iteration
        self.position_history = deque(maxlen=50)
        self.velocity = None  # For compatibility
        self.kpi_logger = kpi_logger
                
        # Internal state
        self.population = None
        self.best_solution = None
        self._rng = np.random.RandomState()
    
    def run(self, visualize_callback: callable = None, kpi_logger=None) -> dict:
        """Optimized DE execution with unified logging"""
        # 🔴 Capture initial state
        original_state = self.env.get_state_snapshot()
        if kpi_logger is None:
            kpi_logger = self.kpi_logger
        print("Using KPI logger:", kpi_logger, flush=True)
            
        self._initialize_population(self.env)
        self.best_solution = self.population[0]
        # self.best_fitness_history = []
        
        for iteration in range(self.iterations):
            self._adapt_parameters(iteration)
            new_population = []
            current_fitness = []  # Track fitness of current population
            best_iter_fitness = -np.inf
            best_iter_metrics = {}  # Track metrics for logging
            
            # Main DE loop
            for i in range(self.population_size):
                trial = self._create_trial_vector(i, self.env)
                trial = self._repair_solution(trial, self.env)
                
                current_metrics = self.env.evaluate_detailed_solution(self.population[i])
                trial_metrics = self.env.evaluate_detailed_solution(trial)
                
                current_fit = current_metrics["fitness"]
                trial_fit = trial_metrics["fitness"]
                
                if trial_fit > current_fit:
                    new_population.append(trial)
                    if trial_fit > best_iter_fitness:
                        best_iter_fitness = trial_fit
                        self.best_solution = trial.copy()
                        best_iter_metrics = trial_metrics  # Track best metrics
                else:
                    new_population.append(self.population[i])
                    if current_fit > best_iter_fitness:
                        best_iter_fitness = current_fit
                        best_iter_metrics = current_metrics  # Track best metrics
            
            # ✅ Unified logging (once per iteration)
            if kpi_logger:
                
                self.kpi_logger.log_metrics(
                    episode=iteration,
                    phase="metaheuristic",
                    algorithm="de",
                    metrics=best_iter_metrics 
                )
                print("Recent KPI Logs:", kpi_logger.recent_logs())
            
            self.population = new_population
            # self.fitness.append(best_iter_fitness)  # Append best fitness of the iteration
            self.fitness[iteration] = best_iter_fitness  # Store fitness directly in the array
            # ✅ Print live updates
            print(f"Iteration {iteration}: Best Fitness = {best_iter_fitness}")
            self.best_fitness_history.append(best_iter_fitness)
            # self._update_visual_state(self.env)
            
            # # Visualization handling (no duplicate logging)
            # if iteration % 5 == 0 and visualize_callback:
            #     print(" Logged Metrics:", kpi_logger.recent_logs()) 
            #     visualize_callback({
            #         "positions": self.positions.tolist(),
            #         "fitness": self.fitness.tolist(),
            #         "algorithm": "de",
            #         "env_state": self.env.get_current_state()
            #     })
                
            #     print(f"DE Visual Update @ Iter {iteration}")
            
            # Environment agent tracking
            self.env.current_metaheuristic_agents = [{
                "position": pos.tolist()[:2],
                "fitness": float(fit),
                "algorithm": "de"
            } for pos, fit in zip(self.positions, self.fitness)]
            
        # 🔴 Restore environment after optimization
        self.env.set_state_snapshot(original_state)    
        self.env.apply_solution(self.best_solution)
            
            # Convert best_solution (a numpy array) to the expected dict format:
        actions = {
                f"bs_{bs_id}": np.where(self.best_solution == bs_id)[0].tolist()
                for bs_id in range(self.env.num_bs)
            }
            # Critical: Update environment state
        # self.env.step(actions)  # Processes agent positions into network state
            
        return {
            "solution": self.best_solution,
            "metrics": self.env.evaluate_detailed_solution(self.best_solution),
            "agents": {
                "positions": self.positions.tolist(),
                "fitness": self.fitness.tolist(),
                "algorithm": "de"
            }
        }
        

    def _initialize_population(self, env: NetworkEnvironment):
        """Generate random UE-BS associations"""
        self.population = [
            self._rng.randint(0, int(env.num_bs), int(env.num_ue))
            for _ in range(int(self.population_size))
        ]
        x_min, y_min = 0, 0
        x_max, y_max = 100, 100 # env.width, env.height
        self.positions = np.random.uniform([x_min, y_min], [x_max, y_max], (self.population_size, 2))

    def _create_trial_vector(self, target_idx: int, env: NetworkEnvironment) -> np.ndarray:
        """DE mutation and crossover operations"""
        candidates = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = self._rng.choice(candidates, 3, False)
        
        mutant = np.clip(
            self.population[a] + self.F * (self.population[b] - self.population[c]),
            0, int(env.num_bs) - 1
        ).astype(int)
        
        trial = self.population[target_idx].copy()
        cross_mask = self._rng.rand(env.num_ue) < self.CR
        trial[cross_mask] = mutant[cross_mask]
        
        return trial

    
    def _repair_solution(self, trial: np.ndarray, env: NetworkEnvironment) -> np.ndarray:
        """Ensure solutions respect BS capacity constraints"""
        bs_counts = np.bincount(trial, minlength=env.num_bs)
        
        # 1. Get capacities for ALL base stations first
        capacities = np.array([int(bs.capacity) for bs in env.base_stations])  # Force to int
        
        # 2. Find overloaded BS indices using vectorized comparison
        overloaded_bs_ids = np.where(bs_counts > capacities)[0]
        
        # 3. Process overloaded BSs
        for bs_id in overloaded_bs_ids:
            excess = int(bs_counts[bs_id] - capacities[bs_id])
            ue_indices = np.where(trial == bs_id)[0]
            np.random.shuffle(ue_indices)
            
            for idx in ue_indices[:excess]:
                available_bs = [
                    b.id for b in env.base_stations 
                    if bs_counts[b.id] < b.capacity
                ]
                if available_bs:
                    new_bs = np.random.choice(available_bs)
                    trial[idx] = new_bs
                    bs_counts[bs_id] -= 1
                    bs_counts[new_bs] += 1
                    
        return trial

    def _adapt_parameters(self, iteration: int):
        """Self-adaptive parameter adjustment"""
        progress = iteration / self.iterations
        self.F = 0.8 - 0.3 * progress  # Linear decay
        self.CR = 0.9 - 0.2 * (1 - progress)  # Curved adaptation
        
    def _calculate_visual_positions(self, env: NetworkEnvironment):
        """Project population to 2D feature space"""
        visual_positions = []
        original_state = env.get_state_snapshot()  # Backup environment state
        
        try:
            # Get BS positions upfront
            bs_positions = {bs.id: bs.position.numpy() for bs in env.base_stations}
            
            for solution in self.population:
                # Feature 1: Load balance (std of UE counts per BS)
                counts = np.bincount(solution, minlength=env.num_bs)
                x = np.std(counts)
                
                # Feature 2: Average SINR (temporary state)
                env.apply_solution({
                    bs_id: np.where(solution == bs_id)[0].tolist() 
                    for bs_id in range(env.num_bs)
                })
                y = np.mean([ue.sinr.item() for ue in env.ues])
                                
                fitness = env.evaluate_detailed_solution(solution)["fitness"]
                
                # Use x (load balance) and y (SINR) as coordinates
                visual_positions.append([x, y, fitness])  # [x, y, fitness]
                
        finally:
            env.set_state_snapshot(original_state)  # Restore environment
        
        return np.array(visual_positions, dtype=np.float32)
    
    def _update_visual_state(self, env: NetworkEnvironment):
        """Enhanced visualization state preparation with Colab compatibility"""
        # Convert solution to spatial coordinates with error handling
        try:
            bs_positions = {bs.id: bs.position.numpy() for bs in env.base_stations}
            solution_metrics = env.evaluate_detailed_solution(self.best_solution)
            current_fitness = solution_metrics["fitness"]
            
            # Create position array with proper dtype for Plotly
            # self.positions = np.array([
            #     [
            #         float(bs_positions[bs_id][0].item()),  # X: BS position
            #         float(bs_positions[bs_id][1].item()),  # Y: BS position
            #         np.float32(current_fitness)      # Z: Fitness value
            #     ]
            #     for bs_id in self.best_solution
            # ], dtype=np.float32)
            
            self.positions = self._calculate_visual_positions(env)
            # Update environment's visualization agents
            env.current_metaheuristic_agents = [
                {"position": pos[:2].tolist(), "fitness": pos[2].item()}
                for pos in self.positions
            ]
            
            # Maintain fitness history with fixed-size buffer
            self.fitness = np.roll(self.fitness, -1)
            self.fitness[-1] = current_fitness

        except KeyError as e:
            print(f"Visualization error - missing BS position: {e}")
        except AttributeError as e:
            print(f"Visualization state error: {e}")



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
