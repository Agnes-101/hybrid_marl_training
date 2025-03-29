# algorithms/de.py
import numpy as np
from collections import deque
from envs.custom_channel_env import NetworkEnvironment

class DEOptimization:
    def __init__(self):
        """Differential Evolution for 6G UE-BS Association"""
        # Optimization parameters
        
        self.population_size = 30
        self.iterations = 50
        self.F = 0.8  # Mutation factor
        self.CR = 0.9  # Crossover rate
        
        # State tracking
        self.positions = np.empty((0, 3))  # Proper initialization
        self.best_fitness_history = []  # Track best fitness per iteration
        self.position_history = deque(maxlen=50)
        self.velocity = None  # For compatibility
        # self.fitness = np.full(100, -np.inf, dtype=np.float32)  # Fixed-size buffer
        # self.position_history = deque(maxlen=50)  # Track movement patterns
        # self.velocity = None  # For compatibility with PSO-based visualizations
        
        # Internal state
        self.population = None
        self.best_solution = None
        self._rng = np.random.RandomState()
        
    def run(self, env: NetworkEnvironment, visualize_callback: callable = None) -> dict:
        """Optimized DE execution with reliable visualization"""
        self._initialize_population(env)
        self.best_solution = self.population[0]
        self.best_fitness_history = []
        self.fitness = np.zeros(self.iterations, dtype=np.float32)
        
        for iteration in range(self.iterations):
            self._adapt_parameters(iteration)
            new_population = []
            best_iter_fitness = -np.inf
            
            # Main DE loop
            for i in range(self.population_size):
                trial = self._create_trial_vector(i, env)
                trial = self._repair_solution(trial, env)
                
                current_fit = env.evaluate_detailed_solution(self.population[i])["fitness"]
                trial_fit = env.evaluate_detailed_solution(trial)["fitness"]
                
                # if trial_fit > current_fit:
                #     new_population.append(trial)
                #     if trial_fit > self.fitness[iteration]:
                #         self.best_solution = trial.copy()
                #         self.fitness[iteration] = trial_fit
                # else:
                #     new_population.append(self.population[i])
                if trial_fit > current_fit:
                    new_population.append(trial)
                    if trial_fit > best_iter_fitness:
                        best_iter_fitness = trial_fit
                        self.best_solution = trial.copy()
                else:
                    new_population.append(self.population[i])
                    if current_fit > best_iter_fitness:
                        best_iter_fitness = current_fit
            
            self.population = new_population
            self.best_fitness_history.append(best_iter_fitness)
            self._update_visual_state(env)  # Update positions based on population
            
                        
            # Unified visualization trigger every 5 iterations
            if iteration % 5 == 0 and visualize_callback:
                visualize_callback({
                    "positions": self.positions.tolist(),
                    "fitness": self.best_fitness_history,# self.fitness[:iteration+1].tolist(),
                    "algorithm": "de"
                })
                print(f"DE VISUAL UPDATE @ Iter {iteration}", flush=True)  # Force output
                # Get metrics from the current best solution
                current_metrics = env.evaluate_detailed_solution(self.best_solution)
                self.kpi_logger.log_kpis(
                    episode=iteration,
                    reward=current_metrics.get("fitness", 0),  # Or whichever key represents reward
                    sinr=current_metrics.get("average_sinr", 0),
                    fairness=current_metrics.get("fairness", 0),
                    load_variance=current_metrics.get("load_variance", 0))
                
            # Environment agent tracking
            env.current_metaheuristic_agents = [{
                "position": pos.tolist()[:2],  # Only x,y for 2D projection
                "fitness": float(fit)
            } for pos, fit in zip(self.positions, self.fitness)]
        
        # Final debug output
        print("\nDE FINAL POSITIONS:", self.positions[:2].tolist())
        print("DE TOP FITNESS:", self.fitness[:5].tolist())
        
        return {
            "solution": self.best_solution,
            "metrics": env.evaluate_detailed_solution(self.best_solution),
            "agents": {
                "positions": self.positions.tolist(),
                "fitness": self.fitness.tolist(),
                "algorithm": "de"
            }
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
        
        # 1. Get capacities for ALL base stations first
        capacities = np.array([bs.capacity for bs in env.base_stations])
        
        # 2. Find overloaded BS indices using vectorized comparison
        overloaded_bs_ids = np.where(bs_counts > capacities)[0]
        
        # 3. Process overloaded BSs
        for bs_id in overloaded_bs_ids:
            excess = bs_counts[bs_id] - capacities[bs_id]
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
        """Convert population to 2D visualization coordinates"""
        visual_positions = []
        # Preserve original environment state to avoid side effects
        original_state = env.get_state()
        
        for solution in self.population:
            # Feature 1: Load balance (std of UE counts per BS)
            counts = np.bincount(solution, minlength=env.num_bs)
            x = np.std(counts)
            
            # Feature 2: Average SINR
            # env.apply_solution(self._vector_to_solution(solution, env))            
            try:
                # Apply solution using dictionary comprehension
                env.apply_solution({
                    bs_id: np.where(solution == bs_id)[0].tolist() 
                    for bs_id in range(env.num_bs)
                })
                y = np.mean([ue.sinr.item() for ue in env.users])
                
            finally:
                env.set_state(original_state)  # Always restore state
            
            # Get current fitness for coloring
            fitness = env.evaluate_detailed_solution(solution)["fitness"]
            visual_positions.append([x, y, fitness])
        
        return np.array(visual_positions, dtype=np.float32)

    # def _update_visual_state(self, env: NetworkEnvironment):
    #     """Prepare universal visualization state"""
    #     # Convert best solution to spatial coordinates
    #     bs_positions = {bs.id: bs.position.numpy() for bs in env.base_stations}
    #     current_fitness = env.evaluate_detailed_solution(self.best_solution)["fitness"]
        
    #     # Update positions (BS x, BS y, current fitness)
    #     self.positions = np.array([
    #         [bs_positions[bs_id][0], bs_positions[bs_id][1], current_fitness]
    #         for bs_id in self.best_solution
    #     ])
        
    #     # Maintain rolling fitness history (fixed-size array)
    #     num_to_add = max(0, self.population_size - len(self.fitness))
    #     self.fitness = np.concatenate([
    #         self.fitness,
    #         np.full(num_to_add, current_fitness)  # No negative values
    #     ])[-self.population_size:]  # Keep LAST N entries (most recent)
    
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
