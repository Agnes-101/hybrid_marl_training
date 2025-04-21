import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class HarmonySearchOptimization:
    
    def __init__(self, env: NetworkEnvironment, kpi_logger=None):
        self.env=env
        self.num_users = env.num_ue
        self.num_cells = env.num_bs
        self.memory_size = 30
        self.iterations = 20
        self.HMCR = 0.9
        self.PAR = 0.3
        self.seed = 42
        self.rng = np.random.RandomState(self.seed)
        # Initialize harmony memory using the seeded RNG
        self.harmony_memory = [self.rng.randint(0, self.num_cells, size=self.num_users) for _ in range(self.memory_size)]
        self.best_harmony = None
        self.kpi_logger = kpi_logger
        # Visualization states
        self.positions = np.empty((0, 3))  # (diversity, harmony_quality, fitness)
        self.fitness_history = []
        
    def fitness(self, solution):
        return self.env.evaluate_detailed_solution( solution)["fitness"]
    
    def run(self, visualize_callback: callable = None, kpi_logger=None) -> dict:        
        # 🔴 Capture initial state
        original_state = self.env.get_state_snapshot()
        self.best_harmony = max(self.harmony_memory, key=self.fitness)
        for iteration in range(self.iterations):
            # Generate new harmony
            new_harmony = self._improvise_harmony()
            
            # Update harmony memory
            self._update_memory(new_harmony)
            # new_harmony = np.zeros(self.num_users, dtype=int)
            # for j in range(self.num_users):
            #     if self.rng.rand() < self.HMCR:
            #         # Use seeded RNG to select a value from harmony memory
            #         values = [h[j] for h in self.harmony_memory]
            #         new_harmony[j] = self.rng.choice(values)
            #         if self.rng.rand() < self.PAR:
            #             new_harmony[j] = self.rng.randint(0, self.num_cells)
            #     else:
            #         new_harmony[j] = self.rng.randint(0, self.num_cells)
            # # Identify the worst harmony in the memory
            # fitness_values = [self.fitness(h) for h in self.harmony_memory]
            # worst_index = np.argmin(fitness_values)
            # if self.fitness(new_harmony) > self.fitness(self.harmony_memory[worst_index]):
            #     self.harmony_memory[worst_index] = new_harmony
                
            # Track current best
            current_best = max(self.harmony_memory, key=self.fitness)
            current_metrics = self.env.evaluate_detailed_solution( current_best)
            
            # ✅ DE-style logging
            if self.kpi_logger:
                self.kpi_logger.log_metrics(
                    episode=iteration,
                    phase="metaheuristic",
                    algorithm="hs",
                    metrics=current_metrics
                )
            
            # Update environment state
            if current_metrics["fitness"] > self.fitness(self.best_harmony):
                self.best_harmony = current_best.copy()                
        # 🔴 Restore environment after optimization
        self.env.set_state_snapshot(original_state)    
        self.env.apply_solution(self.best_harmony)
        # self.env.step({
        #         f"bs_{bs_id}": np.where(self.best_harmony == bs_id)[0].tolist()
        #         for bs_id in range(self.env.num_bs) })
            # # ✅ Visualization updates
            # self._update_visualization(iteration)
            # if visualize_callback and iteration % 5 == 0:
            #     visualize_callback({
            #         "positions": self.positions.tolist(),
            #         "fitness": self.fitness_history,
            #         "algorithm": "hs",
            #         "env_state": self.env.get_current_state()
            #     })
        return {
            "solution": self.best_harmony,
            "metrics": self.env.evaluate_detailed_solution( self.best_harmony),
            "agents": {
                "positions": self.positions.tolist(),
                "fitness": self.fitness_history,
                "algorithm": "hs"
            }
        }
        
    def _improvise_harmony(self):
        """Core harmony improvisation logic"""
        new_harmony = np.zeros(self.num_users, dtype=int)
        for j in range(self.num_users):
            if self.rng.rand() < self.HMCR:
                values = [h[j] for h in self.harmony_memory]
                new_harmony[j] = self.rng.choice(values)
                if self.rng.rand() < self.PAR:
                    new_harmony[j] = self.rng.randint(0, self.num_cells)
            else:
                new_harmony[j] = self.rng.randint(0, self.num_cells)
        return new_harmony

    def _update_memory(self, new_harmony):
        """Memory update with fitness-based replacement"""
        fitness_values = [self.fitness(h) for h in self.harmony_memory]
        worst_index = np.argmin(fitness_values)
        if self.fitness(new_harmony) > fitness_values[worst_index]:
            self.harmony_memory[worst_index] = new_harmony

    def _update_visualization(self, iteration: int):
        """Track harmony memory state in 3D space"""
        # Diversity: standard deviation of solution elements
        diversity = np.std(np.vstack(self.harmony_memory))
        
        # Harmony quality: ratio of best to average fitness
        fitness_values = [self.fitness(h) for h in self.harmony_memory]
        avg_fitness = np.mean(fitness_values)
        best_fitness = np.max(fitness_values)
        harmony_quality = best_fitness / avg_fitness if avg_fitness > 0 else 1
        
        self.positions = np.vstack([self.positions, 
                                [diversity, harmony_quality, best_fitness]])
        self.fitness_history.append(best_fitness)