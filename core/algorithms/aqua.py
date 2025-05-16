# algorithms/aqua.py
import numpy as np
from envs.custom_channel_env import NetworkEnvironment

class AquilaOptimization:
    def __init__(self, env: NetworkEnvironment, iterations=20,pop_size = 30, expansion_factor = 2.0, spiral_density = 0.1,kpi_logger=None):
        """Aquila Optimizer with four hunting strategies"""
        self.env = env
        self.num_ue = env.num_ue
        self.num_bs = env.num_bs
        
        # Hyperparameters
        self.pop_size = pop_size        # Equivalent to ACO's ants
        self.iterations = iterations
        self.expansion_factor = expansion_factor  # For high soar exploration
        self.spiral_density = spiral_density    # Spiral search intensity
        self.seed = 42
        self.kpi_logger = kpi_logger
        
        # State tracking
        self.positions = np.empty((0, 2))  # Maintains ACO-style visualization format
        self.fitness_history = []
        self.best_solution = None
        self.best_fitness = -np.inf
        self.rng = np.random.RandomState(self.seed)

    def run(self, visualize_callback: callable = None, kpi_logger=None) -> dict:
        """Main optimization loop with ACO-compatible interface"""
        original_state = self.env.get_state_snapshot()
        
        # Initialize population with same structure as ACO solutions
        population = self._initialize_population()
        
        for iteration in range(self.iterations):
            # 1. Evaluate all solutions
            fitness_values = np.array([self.env.evaluate_detailed_solution(sol)["fitness"] 
                              for sol in population])
            
            # 2. Update best solution
            current_best_idx = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = population[current_best_idx].copy()
            
            # 3. Aquila hunting strategies
            population = self._aquila_hunting(population, fitness_values, iteration)
            
            # 4. Logging and visualization (matches ACO structure)
            self._update_visualization(iteration)
            current_best_metrics = self.env.evaluate_detailed_solution(self.best_solution)
            
            if self.kpi_logger:
                self.kpi_logger.log_metrics(
                    episode=iteration,
                    phase="metaheuristic",
                    algorithm="aquila",
                    metrics=current_best_metrics
                )
            
            if visualize_callback:
                viz_metrics = {
                    "fitness": current_best_metrics["fitness"],
                    "average_sinr": current_best_metrics["average_sinr"],
                    "fairness": current_best_metrics["fairness"]
                }
                visualize_callback(viz_metrics, self.best_solution)

        # Restore environment state and return results
        self.env.set_state_snapshot(original_state)
        self.env.apply_solution(self.best_solution)
        
        return {
            "solution": self.best_solution,
            "metrics": self.env.evaluate_detailed_solution(self.best_solution),
            "agents": {
                "positions": self.positions.tolist(),
                "fitness": self.fitness_history,
                "algorithm": "aquila"
            }
        }

    def _initialize_population(self):
        """Initialize solutions with same structure as ACO"""
        population = []
        for _ in range(self.pop_size):
            if self.rng.rand() < 0.2:  # 20% heuristic initialization
                sol = np.array([self._find_nearest_bs(ue.position) 
                              for ue in self.env.ues])
            else:
                sol = self.rng.randint(0, self.num_bs, size=self.num_ue)
            population.append(sol)
        return np.array(population)

    def _aquila_hunting(self, population, fitness, iteration):
        """Implement four-phase Aquila hunting strategies"""
        new_population = []
        t = iteration / self.iterations  # Normalized iteration
        
        for i in range(self.pop_size):
            if t < 0.3:  # High soar exploration
                new_sol = self._high_soar(population, population[i], self.best_solution, t)
            elif t < 0.6:  # Contour flight
                new_sol = self._contour_flight(population[i], self.best_solution, t)
            elif t < 0.9:  # Short glide attack
                new_sol = self._short_glide(population[i], self.best_solution, t)
            else:  # Walk attack
                new_sol = self._walk_attack(population[i], self.best_solution)
            
            new_population.append(self._repair_solution(new_sol))
        
        return np.array(new_population)

    def _high_soar(self, population, current, best, t):
        levy = self._levy_flight()
        return (1 - t) * best \
            + t * levy * self.expansion_factor \
                    * (current - np.mean(population, axis=0))

    def _contour_flight(self, current, best, t):
        """Spiral search around best solution"""
        theta = self.rng.uniform(0, 2*np.pi)
        spiral = np.exp(theta * self.spiral_density) * np.cos(theta)
        return best + spiral * (current - best)

    def _short_glide(self, current, best, t):
        """Targeted local search"""
        alpha = self.rng.rand()
        return alpha * best + (1 - alpha) * current + self.rng.normal(0, 0.1, self.num_ue)

    def _walk_attack(self, current, best):
        """Intensive exploitation"""
        return best + self.rng.rand(self.num_ue) * (best - current) / 2.0

    def _repair_solution(self, solution):
        """Identical capacity repair to ACO, with float-safe casting and clipping"""
        # Step 1: Ensure all assignments are integer BS indices in [0, num_bs)
        solution = np.clip(solution.astype(int), 0, self.num_bs - 1)

        # Step 2: Get current load per BS
        counts = np.bincount(solution, minlength=self.num_bs)
        capacities = np.array([bs.capacity for bs in self.env.base_stations])

        # Step 3: For every overloaded BS, reassign its excess users
        overloaded = np.where(counts > capacities)[0]
        for bs in overloaded:
            users = np.where(solution == bs)[0]
            # Compute how many to move, force to int ≥ 0
            raw_excess = counts[bs] - capacities[bs]
            num_excess = max(0, int(raw_excess))

            if num_excess == 0:
                continue

            # Pick that many users (you could also randomize selection here)
            to_move = users[:num_excess]

            # Find new BS indices for each of these users
            new_bs = self._find_alternative_bs(to_move, counts)

            # Reassign and immediately update counts
            solution[to_move] = new_bs
            # decrement old BS, increment new ones
            counts[bs] -= num_excess
            np.add.at(counts, new_bs, 1)

        return solution



    def _find_nearest_bs(self, position):
        """Same helper as ACO"""
        pos = np.array(position)
        bs_positions = np.stack([bs.position for bs in self.env.base_stations])
        dists = np.linalg.norm(bs_positions - pos, axis=1)
        return np.argmin(dists)

    def _find_alternative_bs(self, users, counts):
        """Same capacity helper as ACO"""
        available = np.where(counts < np.array([bs.capacity for bs in self.env.base_stations]))[0]
        if len(available) == 0:
            return self.rng.randint(0, self.num_bs, size=len(users))
        return available[np.argmin(counts[available])]

    def _levy_flight(self):
        """Lévy flight distribution for exploration"""
        beta = 1.5
        sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2) / 
                (np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
        u = self.rng.normal(0, sigma, self.num_ue)
        v = self.rng.normal(0, 1, self.num_ue)
        return 0.01 * u / np.abs(v)**(1/beta)

    def _update_visualization(self, iteration):
        """Maintain ACO-style visualization compatibility"""
        if iteration % 5 == 0:  # Match ACO's visualization frequency
            # Create dummy pheromone-like metrics for compatibility
            bs_counts = np.bincount(self.best_solution, minlength=self.num_bs)
            self.positions = np.vstack([
                self.positions,
                np.column_stack([bs_counts, np.zeros(self.num_bs)])
            ])
        
        self.fitness_history.append(self.best_fitness)