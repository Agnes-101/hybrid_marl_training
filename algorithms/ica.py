import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class ICAOptimization:
    def __init__(self,  env, kpi_logger=None):
        self.env=env
        self.num_users = env.num_ue
        self.num_cells = env.num_bs
        self.population_size = 30
        self.imperialist_count = 5
        self.iterations = 20
        self.seed = 42
        self.rng = np.random.RandomState(self.seed)
        # Initialize population using the seeded RNG
        self.population = [self.rng.randint(0, self.num_cells, size=self.num_users) for _ in range(self.population_size)]
        self.kpi_logger = kpi_logger
        # Visualization states
        self.positions = np.empty((0, 3))  # (imperial_power, diversity, fitness)
        self.fitness_history = []
        self.imperialists = []
        self.colonies = []
        
    def fitness(self, solution):
        return self.env.evaluate_detailed_solution(solution)["fitness"]
    
    def run(self, env: NetworkEnvironment, visualize_callback: callable = None, kpi_logger=None) -> dict:
        # sorted_population = sorted(self.population, key=self.fitness, reverse=True)
        # imperialists = sorted_population[:self.imperialist_count]
        # colonies = sorted_population[self.imperialist_count:]
        # Initial empire formation
        self.population.sort(key=self.fitness, reverse=True)
        self.imperialists = self.population[:self.imperialist_count]
        self.colonies = self.population[self.imperialist_count:]
        
        best_solution = self.imperialists[0].copy()
        best_fitness = self.fitness(best_solution)
        
        for iteration in range(self.iterations):
            # for i in range(len(colonies)):
            #     imp = imperialists[i % self.imperialist_count]
            #     colony = colonies[i].copy()
            #     # Use the seeded RNG for selecting a random index
            #     idx = self.rng.randint(0, self.num_users)
            #     colony[idx] = imp[idx]
            #     if self.fitness(colony) > self.fitness(colonies[i]):
            #         colonies[i] = colony
            new_colonies = []
            for i, colony in enumerate(self.colonies):
                imp = self.imperialists[i % self.imperialist_count]
                new_colony = self._assimilate(colony, imp)
                if self.fitness(new_colony) > self.fitness(colony):
                    new_colonies.append(new_colony)
                else:
                    new_colonies.append(colony)
            self.colonies = new_colonies
            
            self.population = self.imperialists + self.colonies
            # self.population = sorted(self.population, key=self.fitness, reverse=True)            
            self.population.sort(key=self.fitness, reverse=True)
            self.imperialists = self.population[:self.imperialist_count]
            self.colonies = self.population[self.imperialist_count:]
            # Track best solution
            current_best = self.imperialists[0]
            current_metrics = env.evaluate_detailed_solution( current_best)
            
            # ✅ DE-style logging
            if self.kpi_logger:
                self.kpi_logger.log_metrics(
                    episode=iteration,
                    phase="metaheuristic",
                    algorithm="ica",
                    metrics=current_metrics
                )
            
            # Update environment
            if current_metrics["fitness"] > best_fitness:
                best_fitness = current_metrics["fitness"]
                best_solution = current_best.copy()
                self.env.apply_solution(best_solution)
                self.env.step({
                    f"bs_{bs_id}": np.where(best_solution == bs_id)[0].tolist()
                    for bs_id in range(self.env.num_bs)
                })

            # # ✅ Visualization updates
            # self._update_visualization(iteration)
            # if visualize_callback and iteration % 5 == 0:
            #     visualize_callback({
            #         "positions": self.positions.tolist(),
            #         "fitness": self.fitness_history,
            #         "algorithm": "ica",
            #         "env_state": self.env.get_current_state()
            #     })
        return {
            "solution": best_solution,
            "metrics": self.env.evaluate_detailed_solution( best_solution),
            "agents": {
                "positions": self.positions.tolist(),
                "fitness": self.fitness_history,
                "algorithm": "ica"
            }}
            
    def _assimilate(self, colony, imperialist):
        """Core assimilation logic with seeded RNG"""
        new_colony = colony.copy()
        idx = self.rng.randint(0, self.num_users)
        new_colony[idx] = imperialist[idx]
        return new_colony

    def _update_visualization(self, iteration: int):
        """Track imperial competition metrics"""
        # Imperial power: best imperialist's fitness
        imperial_power = self.fitness(self.imperialists[0])
        
        # Diversity: std of all solutions
        all_solutions = self.imperialists + self.colonies
        diversity = np.std(np.vstack(all_solutions))
        
        self.positions = np.vstack([self.positions, 
                                [imperial_power, diversity, imperial_power]])
        self.fitness_history.append(imperial_power)