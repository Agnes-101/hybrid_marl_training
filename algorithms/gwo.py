# algorithms/gwo.py
import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class GWOOptimization:
    def __init__(self,env, kpi_logger=None):
        
        self.env=env
        self.num_users = env.num_ue
        self.num_cells = env.num_bs
        self.swarm_size = 30
        self.iterations = 20
        self.a_initial = 2.0
        self.a_decay = 0.04
        self.seed=42
        self.rng = np.random.RandomState(self.seed)
        
        # Initialize population using RNG
        self.population = [self.rng.randint(0, self.num_cells, size=self.num_users)
                        for _ in range(self.swarm_size)]
        self.alpha = None
        self.beta = None
        self.delta = None
        self.update_leaders()
        self.kpi_logger = kpi_logger
        # Visualization states
        self.positions = np.empty((0, 3))  # (leader_strength, diversity, fitness)
        self.fitness_history = []
        
    def fitness(self, solution):
        return self.env. evaluate_detailed_solution(solution)["fitness"]

    def update_leaders(self):
        sorted_pop = sorted(self.population, key=lambda s: self.fitness(s), reverse=True)
        self.alpha = sorted_pop[0].copy()
        self.beta = sorted_pop[1].copy() if len(sorted_pop) > 1 else sorted_pop[0].copy()
        self.delta = sorted_pop[2].copy() if len(sorted_pop) > 2 else sorted_pop[0].copy()

    def run(self, env: NetworkEnvironment, visualize_callback: callable = None, kpi_logger=None) -> dict:
        for t in range(self.iterations):
            a = self.a_initial - t * self.a_decay  # Use configured parameters
            new_population = []
            for i in range(self.swarm_size):
                new_solution = self.population[i].copy()
                for j in range(self.num_users):
                    # Use RNG instead of np.random
                    r1, r2 = self.rng.rand(), self.rng.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha[j] - self.population[i][j])
                    X1 = self.alpha[j] - A1 * D_alpha
                    
                    r1, r2 = self.rng.rand(), self.rng.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta[j] - self.population[i][j])
                    X2 = self.beta[j] - A2 * D_beta
                    
                    r1, r2 = self.rng.rand(), self.rng.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta[j] - self.population[i][j])
                    X3 = self.delta[j] - A3 * D_delta
                    
                    new_val = int(round((X1 + X2 + X3) / 3))
                    new_solution[j] = np.clip(new_val, 0, self.env.num_cells - 1)
                new_population.append(new_solution)
            self.population = new_population
            self.update_leaders()
            # ✅ DE-style logging
            current_metrics = env.evaluate_detailed_solution(self.alpha)
            if self.kpi_logger:
                self.kpi_logger.log_metrics(
                    episode=t,
                    phase="metaheuristic",
                    algorithm="gwo",
                    metrics=current_metrics
                )
            
            # ✅ Environment update
            self.env.apply_solution(self.alpha)
            self.env.step({
                f"bs_{bs_id}": np.where(self.alpha == bs_id)[0].tolist()
                for bs_id in range(self.env.num_bs)
            })
            
            # ✅ Visualization updates
            # self._update_visualization(t)
            # if visualize_callback and t % 5 == 0:
            #     visualize_callback({
            #         "positions": self.positions.tolist(),
            #         "fitness": self.fitness_history,
            #         "algorithm": "gwo",
            #         "env_state": self.env.get_current_state()
            #     })
            
        return {
            "solution": self.alpha,
            "metrics": self.env.evaluate_detailed_solution( self.alpha),
            "agents": {
                "positions": self.positions.tolist(),
                "fitness": self.fitness_history,
                "algorithm": "gwo"
            }
        }
        
    def _update_visualization(self, iteration: int):
        """Track swarm state in 3D space compatible with DE/PFO"""
        # Leader strength: fitness ratio between alpha and average
        fitness_values = [self.fitness(wolf) for wolf in self.population]
        avg_fitness = np.mean(fitness_values)
        leader_strength = self.fitness(self.alpha) / avg_fitness if avg_fitness > 0 else 1
        
        # Diversity: mean pairwise hamming distance
        diversity = np.mean([np.sum(a != b) for a in self.population for b in self.population])
        
        # Best fitness
        best_fitness = self.fitness(self.alpha)
        
        self.positions = np.vstack([self.positions, [leader_strength, diversity, best_fitness]])
        self.fitness_history.append(best_fitness)    