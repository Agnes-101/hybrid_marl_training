import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class TabuSearchOptimization:
    def __init__(self,  env, kpi_logger=None):
        self.env=env
        self.num_users = env.num_ue
        self.num_cells = env.num_bs
        self.iterations = 20
        self.tabu_size = 10
        self.seed = 42
        self.kpi_logger = kpi_logger
        self.rng = np.random.RandomState(self.seed)
        # Initialize current solution using the seeded RNG
        self.current_solution = self.rng.randint(0, self.num_cells, size=self.num_users)
        self.tabu_list = []
        # Visualization states
        self.positions = np.empty((0, 3))  # (tabu_usage, diversity, fitness)
        self.fitness_history = []
        # Initialize search state
        self.current_solution = self.rng.randint(0, self.num_cells, self.num_users)
        self.best_solution = self.current_solution.copy()
    
    def fitness(self, solution):
        return self.env.evaluate_detailed_solution( solution)["fitness"]
    
    def run(self, env: NetworkEnvironment, visualize_callback: callable = None, kpi_logger=None) -> dict:
        
        best_fitness = self.fitness(self.best_solution)
        for iteration in range(self.iterations):
            # neighbors = []
            # for i in range(self.num_users):
            #     for new_bs in range(self.num_cells):
            #         if new_bs != self.current_solution[i]:
            #             neighbor = self.current_solution.copy()
            #             neighbor[i] = new_bs
            #             if tuple(neighbor) not in self.tabu_list:
            #                 neighbors.append(neighbor)
            # if not neighbors:
            #     break
            # neighbor_fitness = [self.fitness(n) for n in neighbors]
            # best_neighbor = neighbors[np.argmax(neighbor_fitness)]
            # current_fitness = self.fitness(best_neighbor)
            # Generate non-tabu neighbors
            neighbors = self._generate_neighbors()
            if not neighbors:
                break  # No valid neighbors available
                
            # Select best neighbor
            neighbor_fitness = [self.fitness(n) for n in neighbors]
            best_neighbor = neighbors[np.argmax(neighbor_fitness)]
            current_fitness = max(neighbor_fitness)
            
            # Update current solution
            self.current_solution = best_neighbor.copy()
            self._update_tabu(best_neighbor)
            
            # Track best solution
            if current_fitness > best_fitness:
                self.best_solution = best_neighbor.copy()
                best_fitness = current_fitness        
                
            # self.current_solution = best_neighbor.copy()
            # self.tabu_list.append(tuple(self.current_solution))
            # if len(self.tabu_list) > self.tabu_size:
            #     self.tabu_list.pop(0)
            
            # ✅ DE-style logging
            current_metrics = env.evaluate_detailed_solution(self.best_solution)
            if self.kpi_logger:
                self.kpi_logger.log_metrics(
                    episode=iteration,
                    phase="metaheuristic",
                    algorithm="tabu",
                    metrics=current_metrics
                )
            
            # ✅ Environment update
            self.env.apply_solution(self.best_solution)
            self.env.step({
                f"bs_{bs_id}": np.where(self.best_solution == bs_id)[0].tolist()
                for bs_id in range(self.env.num_bs)
            })

            # ✅ Visualization updates
            # self._update_visualization(iteration)
            # if visualize_callback and iteration % 5 == 0:
            #     visualize_callback({
            #         "positions": self.positions.tolist(),
            #         "fitness": self.fitness_history,
            #         "algorithm": "tabu",
            #         "env_state": self.env.get_current_state()
            #     })

        return {
            "solution": self.best_solution,
            "metrics": self.env.evaluate_detailed_solution( self.best_solution),
            "agents": {
                "positions": self.positions.tolist(),
                "fitness": self.fitness_history,
                "algorithm": "tabu"
            }
        }
            
            
    def _generate_neighbors(self):
        """Generate all valid non-tabu neighbors"""
        neighbors = []
        for i in range(self.num_users):
            for new_bs in range(self.num_cells):
                if new_bs != self.current_solution[i]:
                    neighbor = self.current_solution.copy()
                    neighbor[i] = new_bs
                    if tuple(neighbor) not in self.tabu_list:
                        neighbors.append(neighbor)
        return neighbors

    def _update_tabu(self, solution):
        """Manage tabu list with FIFO policy"""
        self.tabu_list.append(tuple(solution))
        if len(self.tabu_list) > self.tabu_size:
            self.tabu_list.pop(0)

    def _update_visualization(self, iteration: int):
        """Track search progress in 3D space"""
        # Tabu usage: percentage of tabu list capacity used
        tabu_usage = len(self.tabu_list) / self.tabu_size
        
        # Diversity: unique solutions in tabu list
        diversity = len(set(self.tabu_list)) / self.tabu_size
        
        # Best fitness
        best_fitness = self.fitness(self.best_solution)
        
        self.positions = np.vstack([self.positions, [tabu_usage, diversity, best_fitness]])
        self.fitness_history.append(best_fitness)    
