import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class AVOAOptimization:
    def __init__(self, env: NetworkEnvironment, kpi_logger=None):
        """African Vultures Optimization Algorithm for 6G user association"""
        self.env = env
        self.vultures = 30     # Population size
        self.iterations = 20
        self.exploration_rate = 0.5   # Initial exploration probability
        self.exploitation_rate = 0.8  # Exploitation intensity
        self.satiety_threshold = 0.3  # Food scarcity threshold
        self.alpha = 2.0       # Exploration random walk coefficient
        self.beta = 1.5        # Exploitation convergence coefficient
        self.seed = 42
        self.kpi_logger = kpi_logger
        
        # Visualization and tracking
        self.positions = np.empty((0, 2))  # (mean_assoc, std_assoc)
        self.fitness_history = []
        self.rng = np.random.RandomState(self.seed)
        self.best_solution = None
        self.second_best_solution = None

    def run(self, visualize_callback=None, kpi_logger=None) -> dict:
        """Main optimization process"""
        original_state = self.env.get_state_snapshot()
        num_ue = self.env.num_ue
        num_bs = self.env.num_bs
        
        # Initialize vulture population
        solutions = np.array([self._random_solution(num_ue, num_bs) 
                            for _ in range(self.vultures)])
        best_fitness = -np.inf
        self.best_solution = solutions[0].copy()
        self.second_best_solution = solutions[1].copy()

        for iteration in range(self.iterations):
            # Adaptive parameter adjustment
            satiety = 1 - (iteration / self.iterations)
            exploration_prob = self.exploration_rate * satiety
            
            # Evaluate fitness
            fitness_values = np.array([self.env.evaluate_detailed_solution(sol)["fitness"] 
                                    for sol in solutions])
            sorted_indices = np.argsort(-fitness_values)
            
            # Update best solutions
            self.best_solution = solutions[sorted_indices[0]].copy()
            self.second_best_solution = solutions[sorted_indices[1]].copy()
            current_best_fitness = fitness_values[sorted_indices[0]]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness

            # Update positions
            solutions = self._update_vultures(solutions, satiety, exploration_prob)
            
            # Logging and visualization
            # self._update_visualization(iteration)
            current_metrics = self.env.evaluate_detailed_solution(self.best_solution)
            
            if self.kpi_logger:
                self.kpi_logger.log_metrics(
                    episode=iteration,
                    phase="metaheuristic",
                    algorithm="avoa",
                    metrics=current_metrics
                )
                print(f"AVOA Iter {iteration}: Best Fitness = {best_fitness:.4f}")

            if visualize_callback:
                viz_metrics = {
                    "fitness": current_metrics["fitness"],
                    "average_sinr": current_metrics["average_sinr"],
                    "fairness": current_metrics["fairness"]
                }
                visualize_callback(viz_metrics, self.best_solution)

        # Finalize and return results
        self.env.set_state_snapshot(original_state)
        self.env.apply_solution(self.best_solution)
        
        return {
            "solution": self.best_solution,
            "metrics": self.env.evaluate_detailed_solution(self.best_solution),
            "agents": {
                "positions": self.positions.tolist(),
                "fitness": self.fitness_history,
                "algorithm": "avoa"
            }
        }

    def _random_solution(self, num_ue: int, num_bs: int) -> np.ndarray:
        """Generate random UE-BS associations"""
        return self.rng.randint(0, num_bs, size=num_ue)

    def _update_vultures(self, solutions: np.ndarray, satiety: float, exploration_prob: float) -> np.ndarray:
        """Update vulture positions based on scavenging behavior"""
        new_solutions = []
        num_bs = self.env.num_bs
        
        for idx, sol in enumerate(solutions):
            if self.rng.rand() < exploration_prob:
                # Exploration phase: random search for new carcasses
                new_sol = self._exploration_phase(sol)
            else:
                # Exploitation phase: compete over existing resources
                if self.rng.rand() < self.satiety_threshold:
                    new_sol = self._intense_exploitation(sol)
                else:
                    new_sol = self._mild_exploitation(sol, satiety)
            new_solutions.append(new_sol)
            
        return np.array(new_solutions)

    def _exploration_phase(self, current_sol: np.ndarray) -> np.ndarray:
        """Random search with Levy flight characteristics"""
        new_sol = current_sol.copy()
        num_ue = len(new_sol)
        
        # Generate Levy flight steps
        levy_step = self._levy_flight()
        
        for ue in range(num_ue):
            if self.rng.rand() < 0.5:
                # Random jump with Levy distribution
                new_bs = int((current_sol[ue] + levy_step) % self.env.num_bs)
                new_sol[ue] = new_bs
        return new_sol

    def _intense_exploitation(self, current_sol: np.ndarray) -> np.ndarray:
        """Aggressive competition between best vultures"""
        new_sol = current_sol.copy()
        num_ue = len(new_sol)
        
        for ue in range(num_ue):
            # Follow either best or second-best solution
            if self.rng.rand() < 0.5:
                leader = self.best_solution
            else:
                leader = self.second_best_solution
                
            # Calculate movement toward leader
            delta = leader[ue] - current_sol[ue]
            step = int(self.beta * delta * self.rng.rand())
            new_sol[ue] = (current_sol[ue] + step) % self.env.num_bs
        return new_sol

    def _mild_exploitation(self, current_sol: np.ndarray, satiety: float) -> np.ndarray:
        """Cooperative exploitation around best solution"""
        new_sol = current_sol.copy()
        num_ue = len(new_sol)
        
        for ue in range(num_ue):
            # Calculate spiral movement around best solution
            theta = self.rng.uniform(0, 2*np.pi)
            radius = int(self.alpha * satiety * self.env.num_bs)
            step = int(radius * np.cos(theta))
            new_sol[ue] = (self.best_solution[ue] + step) % self.env.num_bs
        return new_sol

    def _levy_flight(self) -> float:
        """Generate Levy flight step size"""
        beta = 1.5
        sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2) / \
               (np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
        u = self.rng.normal(0, sigma)
        v = self.rng.normal(0, 1)
        step = 0.01 * u / abs(v)**(1/beta)
        return int(step * self.env.num_bs)

    def _update_visualization(self, iteration: int):
        """Track solution diversity metrics"""
        current_solutions = self.env.get_current_associations()
        mean_assoc = np.mean(current_solutions)
        std_assoc = np.std(current_solutions)
        
        self.positions = np.vstack([self.positions, [mean_assoc, std_assoc]])
        self.fitness_history.append(
            self.env.evaluate_detailed_solution(self.best_solution)["fitness"]
        )