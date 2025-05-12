import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class HippoOptimization:
    def __init__(self, env: NetworkEnvironment,iterations=20, kpi_logger=None):
        """Hippopotamus Optimization Algorithm for 6G load balancing"""
        self.env = env
        self.pod_size = 30          # Population size
        self.iterations = iterations
        self.aggression_rate = 0.3  # Territorial defense probability
        self.social_factor = 0.6    # Pod cohesion strength
        self.yawn_impact = 0.4      # Dominance display effect
        self.territorial_decay = 0.9  # Exploration reduction rate
        self.seed = 42
        self.kpi_logger = kpi_logger
        
        # Tracking and visualization
        self.positions = np.empty((0, 2))  # (mean_assoc, std_assoc)
        self.fitness_history = []
        self.rng = np.random.RandomState(self.seed)
        self.alpha_hippo = None     # Dominant solution
        self.territory_center = None

    def run(self, visualize_callback=None, kpi_logger=None) -> dict:
        """Main optimization process"""
        original_state = self.env.get_state_snapshot()
        num_ue = self.env.num_ue
        num_bs = self.env.num_bs
        
        # Initialize hippo pod and territory
        solutions = np.array([self._random_solution(num_ue, num_bs) 
                            for _ in range(self.pod_size)])
        self.territory_center = np.mean(solutions, axis=0).astype(int)
        self.alpha_hippo = solutions[0].copy()
        best_fitness = -np.inf

        for iteration in range(self.iterations):
            # Adaptive parameter adjustment
            current_aggression = self.aggression_rate * (self.territorial_decay ** iteration)
            social_cohesion = self.social_factor * (1 - iteration/self.iterations)
            
            # Update hippo positions
            solutions = self._update_pod(solutions, social_cohesion, current_aggression)
            
            # Evaluate fitness
            fitness_values = np.array([self.env.evaluate_detailed_solution(sol)["fitness"] 
                                    for sol in solutions])
            current_best_idx = np.argmax(fitness_values)
            
            # Update alpha hippo and territory
            if fitness_values[current_best_idx] > best_fitness:
                best_fitness = fitness_values[current_best_idx]
                self.alpha_hippo = solutions[current_best_idx].copy()
                self._update_territory(solutions)

            # Logging and visualization
            # self._update_visualization(iteration)
            current_metrics = self.env.evaluate_detailed_solution(self.alpha_hippo)
            
            if self.kpi_logger:
                self.kpi_logger.log_metrics(
                    episode=iteration,
                    phase="metaheuristic",
                    algorithm="hoa",
                    metrics=current_metrics
                )
                print(f"HOA Iter {iteration}: Best Fitness = {best_fitness:.4f}")

            if visualize_callback:
                viz_metrics = {
                    "fitness": current_metrics["fitness"],
                    "average_sinr": current_metrics["average_sinr"],
                    "fairness": current_metrics["fairness"]
                }
                visualize_callback(viz_metrics, self.alpha_hippo)

        # Finalize and return results
        self.env.set_state_snapshot(original_state)
        self.env.apply_solution(self.alpha_hippo)
        
        return {
            "solution": self.alpha_hippo,
            "metrics": self.env.evaluate_detailed_solution(self.alpha_hippo),
            "agents": {
                "positions": self.positions.tolist(),
                "fitness": self.fitness_history,
                "algorithm": "hoa"
            }
        }

    def _random_solution(self, num_ue: int, num_bs: int) -> np.ndarray:
        """Generate random UE-BS associations"""
        return self.rng.randint(0, num_bs, size=num_ue)

    def _update_pod(self, solutions: np.ndarray, cohesion: float, aggression: float) -> np.ndarray:
        """Update hippo positions based on social and territorial behaviors"""
        new_solutions = []
        num_bs = self.env.num_bs
        
        for idx, sol in enumerate(solutions):
            # Social hierarchy influence
            if self.rng.rand() < self.yawn_impact:
                new_sol = self._dominance_influence(sol)
            else:
                if self.rng.rand() < aggression:
                    new_sol = self._territorial_defense(sol)
                else:
                    new_sol = self._river_movement(sol, cohesion)
            new_solutions.append(new_sol)
            
        return np.array(new_solutions)

    def _dominance_influence(self, current_sol: np.ndarray) -> np.ndarray:
        """Alpha hippo's influence on pod members"""
        new_sol = current_sol.copy()
        num_ue = len(new_sol)
        
        # Blend with alpha's solution
        blend_mask = self.rng.rand(num_ue) < 0.6
        new_sol[blend_mask] = self.alpha_hippo[blend_mask]
        
        # Random challenges to hierarchy
        challenge_mask = self.rng.rand(num_ue) < 0.1
        new_sol[challenge_mask] = self.rng.randint(0, self.env.num_bs, size=sum(challenge_mask))
        
        return new_sol

    def _territorial_defense(self, current_sol: np.ndarray) -> np.ndarray:
        """Local exploitation within territory"""
        new_sol = current_sol.copy()
        num_ue = len(new_sol)
        
        # Stay near territory center with random patrols
        for ue in range(num_ue):
            if self.rng.rand() < 0.7:  # 70% chance to follow territory
                new_sol[ue] = self.territory_center[ue]
            else:  # 30% local exploration
                new_sol[ue] = (current_sol[ue] + self.rng.randint(-1, 2)) % self.env.num_bs
        return new_sol

    def _river_movement(self, current_sol: np.ndarray, cohesion: float) -> np.ndarray:
        """Group movement in water (exploration)"""
        new_sol = current_sol.copy()
        num_ue = len(new_sol)
        
        # Coordinated group movement
        if self.rng.rand() < cohesion:
            group_shift = self.rng.randint(-2, 3)
            new_sol = (current_sol + group_shift) % self.env.num_bs
        else:
            # Individual random exploration
            for ue in range(num_ue):
                if self.rng.rand() < 0.4:
                    new_sol[ue] = self.rng.randint(0, self.env.num_bs)
        return new_sol

    def _update_territory(self, solutions: np.ndarray):
        """Update territory center based on pod distribution"""
        self.territory_center = np.median(solutions, axis=0).astype(int)

    def _update_visualization(self, iteration: int):
        """Track territory metrics for visualization"""
        current_solutions = self.env.get_current_associations()
        mean_assoc = np.mean(current_solutions)
        std_assoc = np.std(current_solutions)
        
        self.positions = np.vstack([self.positions, [mean_assoc, std_assoc]])
        self.fitness_history.append(
            self.env.evaluate_detailed_solution(self.alpha_hippo)["fitness"]
        )