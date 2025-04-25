import sys
import os
# Add project root to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

import torch
import numpy as np
import gymnasium as gym
from ray.rllib.env import EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from typing import Dict, List
from hybrid_trainer.kpi_logger import KPITracker  # Import the KPI logger
class UE:
    def __init__(self, id, position, velocity, demand):
        self.id = int(id)
        self.position = torch.tensor(position, dtype=torch.float32)
        self.velocity = torch.tensor(velocity, dtype=torch.float32)
        self.demand = demand  # Mbps
        self.associated_bs = None
        self.sinr = 0.0
        self.sinr = torch.tensor(0.0)  # Initialize as tensor

    def update_position(self, delta_time=1.0):
        self.position += self.velocity * delta_time + torch.randn(2) * 0.1

class BaseStation:
    def __init__(self, id, position, frequency, bandwidth):
        self.id = int(id)
        self.position = torch.tensor(position, dtype=torch.float32)
        self.frequency = torch.tensor(frequency, dtype=torch.float32)
        self.bandwidth = torch.tensor(bandwidth , dtype=torch.float32) # MHz
        self.allocated_resources = {}  # {ue_id: allocated_bandwidth}
        self.load = 0.0
        self.capacity = bandwidth // 200  # 200 Mbps/UE

    def calculate_load(self):
        self.load = sum(self.allocated_resources.values()) / self.bandwidth

class NetworkEnvironment(MultiAgentEnv):
    def __init__(self, config:EnvContext, log_kpis=True):        
        super().__init__()  # ✅ Initialize gym.Env
        self.config = config
        # Define observation space first
        # Access passed environment instance
        self.num_bs = config.get("num_bs", 20)
        self.num_ue = config.get("num_ue", 200)
        self.episode_length = config.get("episode_length", 100)
        self.env_instance = config.get("environment_instance")
        
        self.version = 0  # Internal state version
        self.current_step = 0
        self.log_kpis = log_kpis        
        self.metaheuristic_agents = []  # Initialize empty list         
        # Calculate grid dimensions for base stations
        num_bs = self.num_bs
        a = int(np.floor(np.sqrt(num_bs)))
        rows, cols = 1, num_bs  # Default to 1 row if no factors found
        while a > 0:
            if num_bs % a == 0:
                rows = a
                cols = num_bs // a
                break
            a -= 1
        
        # Calculate spacing to avoid edges
        x_spacing = 100.0 / (cols + 1)
        y_spacing = 100.0 / (rows + 1)
        
        # Generate grid positions
        positions = []
        for i in range(rows):
            for j in range(cols):
                x = (j + 1) * x_spacing
                y = (i + 1) * y_spacing
                positions.append([x, y])
                if len(positions) == num_bs:
                    break  # Exit early if we've reached the desired number
            if len(positions) == num_bs:
                break
        
        self.base_stations = [
            BaseStation(id=i, position=positions[i],
                        frequency=100.0, bandwidth=1000.0)
            for i in range(self.num_bs)
        ]
        # self.base_stations = [
        #     BaseStation(id=i, position=[np.random.uniform(0, 100), np.random.uniform(0, 100)],
        #                 frequency=100.0, bandwidth=1000.0)
        #     for i in range(self.num_bs)
        # ]
        self.ues = [
            UE(id=i, position=[np.random.uniform(0, 100), np.random.uniform(0, 100)],
            velocity=[np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
            demand=np.random.randint(50, 200))
            for i in range(self.num_ue)
        ]
        self.associations = {bs.id: [] for bs in self.base_stations}
        # Initialize KPI logger if logging is enabled
        self.kpi_logger = KPITracker() if log_kpis else None
        # Define observation space with explicit float32 dtype
        # # Observation space for each BS agent 
        # self.observation_space = gym.spaces.Dict({
        #     f"bs_{i}": gym.spaces.Box(
        #         low=-np.inf, 
        #         high=np.inf,
        #         shape=(5,),  # Simplified observation: [load, x, y, num_associated_ues, avg_demand]
        #         dtype=np.float32
        #     ) for i in range(self.num_bs)
        # })
                
        # # Action space for each BS agent (UE selection mask)        
        # self.action_space = gym.spaces.Dict({
        #     f"bs_{i}": gym.spaces.Discrete(self.num_ue + 1)  # Action = UE index to associate (0 = no action)
        #     for i in range(self.num_bs)
        # })
        
        # For each UE, observe:
        # - SINR to all BSs (vector of size num_bs)
        # - Load of all BSs (vector of size num_bs)
        # - Own demand (scalar)
        self.observation_space = gym.spaces.Dict({
            f"ue_{i}": gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(2 * self.num_bs + 1,)  # SINRs + BS loads + own demand
                # dtype=np.float32
            ) for i in range(self.num_ue)
        })

        # Action space: UE chooses a BS (Discrete(num_bs))
        self.action_space = gym.spaces.Dict({
            f"ue_{i}": gym.spaces.Discrete(self.num_bs) 
            for i in range(self.num_ue)
        })
        
    def observation_space(self, agent):
        return self.observation_space[agent]

    def action_space(self, agent):
        return self.action_space[agent]

    def reward(self, agent):
        return self.calculate_individual_reward(agent)  # Implement per-BS reward

    def calculate_individual_reward(self, bs_id):
        bs = next(bs for bs in self.base_stations if bs.id == bs_id)
        return (bs.capacity - bs.load) * 0.1 + np.mean([ue.sinr for ue in self.ues if ue.associated_bs == bs_id])
    
    # def reset(self, seed=None, options=None):
    #         # Called automatically by RLlib at episode start
    #     self.current_step = 0
    #     self.version += 1
            
    #         # Reset UE positions and associations
    #     for ue in self.ues:
    #         ue.position = torch.tensor(
    #                 [np.random.uniform(0, 100), np.random.uniform(0, 100)],
    #                 dtype=torch.float32  # Match observation dtype
    #             )
    #         ue.associated_bs = None
            
    #         # Generate initial observations
    #         obs = self._get_obs()
            
    #         # Convert all values to float32 explicitly
    #     return {
    #             agent_id: obs[agent_id].astype(np.float32)
    #             for agent_id in obs
    #         }, {}
    
    def reset(self, seed=None, options=None):
        self.current_step = 0
        # Reset UE positions/associations
        for ue in self.ues:
            ue.position = np.random.uniform(0, 100, size=2)  # Use numpy, not PyTorch
            ue.associated_bs = None
        
        # Return observations for ALL UE agents
        return self._get_obs(), {}

    def calculate_sinr(self, ue, bs):
        distance = torch.norm(ue.position - bs.position)
        path_loss = 32.4 + 20 * torch.log10(distance + 1e-6) + 20 * torch.log10(bs.frequency)
        tx_power = 30  # dBm
        noise_floor = -174 + 10 * torch.log10(bs.bandwidth * 1e6)
        sinr_linear = 10 ** ((tx_power - path_loss - noise_floor) / 10)
        return 10 * torch.log10(sinr_linear + 1e-10)
    
    # def calculate_sinr(self,ue,bs):
    #     if ue.associated_bs is None:
    #         return 0.0
        
    #     bs = self.base_stations[ue.associated_bs]
    #     distance = np.linalg.norm(ue.position - bs.position)
    #     interference = sum(
    #         1.0 / (np.linalg.norm(ue.position - other_bs.position) + 1e-6)
    #         for other_bs in self.base_stations 
    #         if other_bs.id != bs.id
    #     )
    #     return (30.0 - 20 * np.log10(distance)) / (interference + 1e-6)

    # Add these to your NetworkEnvironment class
    def calculate_jains_fairness(self):
        throughputs = [ue.throughput for ue in self.ues]
        return (sum(throughputs) ** 2) / (len(throughputs) * sum(t**2 for t in throughputs))

    @property
    def throughput(self):
        return torch.log2(1 + 10**(self.sinr/10)).item()
        
    def calculate_reward(self):
        """Calculate reward with tensor-safe operations"""
        # Convert SINR values to tensor first
        sinr_tensor = torch.tensor([ue.sinr for ue in self.ues], dtype=torch.float32)
        
        throughput = torch.sum(torch.log2(1 + 10**(sinr_tensor/10)))
        loads = torch.tensor([bs.load for bs in self.base_stations])
        
        jain = (loads.sum()**2) / (self.num_bs * (loads**2).sum() + 1e-6)
        overload_penalty = torch.sum(torch.relu(loads - 1.0))
        
        return throughput + 2.0 * jain - 0.5 * overload_penalty
    
    def _admit_nearest_ue(self, bs):
        """Find and associate nearest unassociated UE to BS"""
        # Get all unassociated UEs
        unassociated_ues = [ue for ue in self.ues if ue.associated_bs is None]
        
        if not unassociated_ues:
            return
        
        # Calculate distances using vector math
        bs_pos = bs.position.numpy()
        ue_positions = np.array([ue.position.numpy() for ue in unassociated_ues])
        
        distances = np.linalg.norm(ue_positions - bs_pos, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_ue = unassociated_ues[nearest_idx]
        
        # Associate UE
        nearest_ue.associated_bs = bs.id
        bs.allocated_resources[nearest_ue.id] = nearest_ue.demand

    def _offload_most_demanding_ue(self, bs):
        """Remove UE with highest demand from BS"""
        if not bs.allocated_resources:
            return
        
        # Find UE with max demand
        ue_id = max(bs.allocated_resources, key=lambda k: bs.allocated_resources[k])
        ue = self.ues[ue_id]
        
        # Disassociate
        del bs.allocated_resources[ue_id]
        ue.associated_bs = None

    # def calculate_sinr_vectorized(self, ue, bs):
    #     """Vectorized SINR calculation for a UE-BS pair"""
    #     # Precompute all BS positions
    #     bs_positions = np.array([b.position.numpy() for b in self.base_stations])
        
    #     # Calculate distance to all BSs
    #     distances = np.linalg.norm(bs_positions - ue.position.numpy(), axis=1)
        
    #     # Calculate signal from serving BS
    #     serving_power = bs.transmit_power / (distances[bs.id] ** 2 + 1e-6)
        
    #     # Calculate interference from other BSs
    #     interference = sum(
    #         b.transmit_power / (d ** 2 + 1e-6) 
    #         for b, d in zip(self.base_stations, distances) 
    #         if b.id != bs.id
    #     )
        
    #     # Calculate SINR (handle division by zero)
    #     noise_floor = 1e-10  # Thermal noise
    #     return serving_power / (interference + noise_floor)
        
    def step(self, actions: Dict[str, int]):
        # 1. Process UE actions (BS selections)
        for agent_id, bs_choice in actions.items():
            ue_id = int(agent_id.split("_")[1])
            ue = self.ues[ue_id]
            target_bs = self.base_stations[bs_choice]
            
            # Disassociate from old BS (if any)
            if ue.associated_bs is not None:
                old_bs = self.base_stations[ue.associated_bs]
                if ue.id in old_bs.allocated_resources:
                    del old_bs.allocated_resources[ue.id]
            
            # Associate with new BS if capacity allows
            if target_bs.load + ue.demand <= target_bs.capacity:
                target_bs.allocated_resources[ue.id] = ue.demand
                ue.associated_bs = bs_choice
        
        # 2. Update BS loads and UE SINRs
        for bs in self.base_stations:
            bs.load = sum(bs.allocated_resources.values())
        
        # 3. Calculate rewards (individual + global)
        rewards = {}
        # global_fairness = self.calculate_jains_fairness()  # From earlier code
        for ue in self.ues:
            if ue.associated_bs is None:
                reward = -1.0  # Penalize unconnected UEs
            else:
                bs = self.base_stations[ue.associated_bs]
                throughput = np.log2(1 + 10**(ue.sinr/10))
                # reward = throughput + 0.5 * global_fairness  # Hybrid reward
                reward = self.calculate_individual_reward()
            
            rewards[f"ue_{ue.id}"] = reward
        
        # 4. Check termination
        self.current_step += 1
        done = {"__all__": self.current_step >= self.episode_length}
        
        return self._get_obs(), rewards, done, {}
    
    # def step(self, actions: Dict[str, int]):
    #     self.version += 1
        
    #     # 1. UE Movement (keep existing)
    #     for ue in self.ues:
    #         ue.update_position()

    #     # 2. Process BS Actions with safety checks
    #     for bs_id, action in actions.items():
    #         try:
    #             bs = self.base_stations[int(bs_id.split("_")[1])]
                
    #             # Action 0: No action
    #             if action == 1:
    #                 self._admit_nearest_ue(bs)
    #             elif action == 2:
    #                 self._offload_most_demanding_ue(bs)
                    
    #         except (IndexError, ValueError) as e:
    #             print(f"Invalid action {action} for {bs_id}: {e}")
    #             continue

    #     # 3. Batch Update SINR
    #     associated_pairs = [(ue, ue.associated_bs) for ue in self.ues if ue.associated_bs is not None]
        
    #     # Vectorized SINR calculation
    #     for ue, bs_id in associated_pairs:
    #         bs = self.base_stations[bs_id]
    #         ue.sinr = self.calculate_sinr_vectorized(ue, bs)
        
    #     # 4. Calculate Loads (keep existing)
    #     for bs in self.base_stations:
    #         bs.calculate_load()
        
    #     # 5. Reward and termination
    #     reward = self.calculate_reward()
    #     self.current_step += 1
    #     done = self.current_step >= self.episode_length
        
    #     return self._get_obs(), reward, done, {}
    
    
    
    # def _get_obs(self):
    #     """Normalized BS-centric observations"""
    #     MAX_LOAD = 1000  # Match your BS capacity
    #     MAX_UES = 50     # Max UEs per BS expected
    #     ENV_SIZE = 100   # Match your coordinate system

    #     obs = {}
    #     for bs in self.base_stations:
    #         # Normalized features
    #         obs_key = f"bs_{bs.id}"
    #         obs[obs_key] = np.array([
    #             bs.load / MAX_LOAD,  # Normalized load [0-1]
    #             (bs.position[0] - ENV_SIZE/2) / (ENV_SIZE/2),  # X pos [-1,1]
    #             (bs.position[1] - ENV_SIZE/2) / (ENV_SIZE/2),  # Y pos [-1,1]
    #             len(bs.allocated_resources) / MAX_UES,  # Association count
    #             np.mean(list(bs.allocated_resources.values())) / 200,  # Avg demand
    #             self._calculate_local_interference(bs)  # New metric
    #         ], dtype=np.float32)
        
    #     return obs
    
    def _get_obs(self):
        # Precompute BS loads and positions once
        bs_loads = np.array([bs.load for bs in self.base_stations], dtype=np.float32)
        bs_positions = np.array([bs.position for bs in self.base_stations], dtype=np.float32)
        
        obs = {}
        for ue in self.ues:
            # Calculate SINR to all BSs (vectorized)
            distances = np.linalg.norm(bs_positions - ue.position, axis=1)
            sinrs = 30.0 - 20 * np.log10(distances + 1e-6)  # Simplified model
            
            # Normalize features
            normalized_sinrs = sinrs / 40.0  # Assume max SINR ~40 dB
            normalized_loads = bs_loads / 1000.0  # Assuming max load=1000
            
            obs[f"ue_{ue.id}"] = np.concatenate([
                normalized_sinrs,
                normalized_loads,
                [ue.demand / 200.0]  # Max demand=200
            ], dtype=np.float32)
        
        return obs  # Dict keyed by "ue_0", "ue_1", etc.

    def _calculate_local_interference(self, bs):
        """Calculate interference from neighboring BSs"""
        neighbor_dist = 30  # Consider BSs within 30 units
        interference = 0.0
        
        for other_bs in self.base_stations:
            if other_bs.id == bs.id:
                continue
                
            distance = np.linalg.norm(bs.position - other_bs.position)
            if distance < neighbor_dist:
                interference += other_bs.transmit_power / (distance ** 2 + 1e-6)
        
        return np.log1p(interference)  # Compress scale
        

    def _get_normalized_ue_positions(self, bs):
        """Normalize positions to [-1,1] range"""
        positions = []
        for ue in self.ues:
            rel_pos = (ue.position - bs.position).numpy()
            # Normalize based on environment bounds (0-100 in your case)
            norm_pos = rel_pos / 50.0 - 1.0  # Scales to [-1,1]
            positions.append(norm_pos.astype(np.float32))  # ✅ Cast
        return np.array(positions).flatten()
    
    # In NetworkEnvironment.get_current_state() for visualization
    def get_current_state(self):
        return {
            "base_stations": [
                {
                    "id": bs.id,
                    "position": bs.position.tolist(),  # bs.position.numpy(),
                    "load": bs.load,
                    "capacity": bs.capacity
                } for bs in self.base_stations
            ],
            "users": [
                {
                    "id": ue.id,
                    "position": ue.position.tolist(),  # ue.position.numpy(),
                    "associated_bs": ue.associated_bs,
                    "sinr": ue.sinr.item()  # Convert tensor to float
                } for ue in self.ues
            ],
            "associations": self.associations.copy(),
            "version": self.version,
            # "metaheuristic_agents": self.current_metaheuristic_agents  # Set by optimizer
        }
        
    def get_state_snapshot(self) -> dict:
        """Full state snapshot for rollback support"""
        return {
            # UE state (preserve tensors)
            "users": [{
                "id": ue.id,
                "position": ue.position.clone(),
                "velocity": ue.velocity.clone(),
                "demand": ue.demand,
                "associated_bs": ue.associated_bs,
                "sinr": ue.sinr.clone()
            } for ue in self.ues],
            
            # BS state (preserve allocations)
            "base_stations": [{
                "id": bs.id,
                "allocated_resources": bs.allocated_resources.copy(),
                "load": bs.load,
                "capacity": bs.capacity
            } for bs in self.base_stations],
            
            # Episode tracking
            "current_step": self.current_step
        }

    def set_state_snapshot(self, state: dict):
        """Restore environment from snapshot"""
        # Restore UEs
        for ue_state in state["users"]:
            ue = next(u for u in self.ues if u.id == ue_state["id"])
            ue.position = ue_state["position"].clone()
            ue.velocity = ue_state["velocity"].clone()
            ue.demand = ue_state["demand"]
            ue.associated_bs = ue_state["associated_bs"]
            ue.sinr = ue_state["sinr"].clone()
        
        # Restore Base Stations
        for bs_state in state["base_stations"]:
            bs = next(b for b in self.base_stations if b.id == bs_state["id"])
            bs.allocated_resources = bs_state["allocated_resources"].copy()
            bs.load = bs_state["load"]
            bs.capacity = bs_state["capacity"]
        
        # Restore episode progress
        self.current_step = state["current_step"]
    
    def _update_system_metrics(self):
        """Update all system metrics after applying a new solution"""
        # Update SINR for all users
        for ue in self.ues:
            if ue.associated_bs is not None:
                bs = next(bs for bs in self.base_stations if bs.id == ue.associated_bs)
                ue.sinr = self.calculate_sinr(ue, bs)
        
        # Update load for all base stations
        for bs in self.base_stations:
            bs.calculate_load()
    
    def apply_solution(self, solution):
        """Apply a solution (numpy array or dict) to the environment"""
        # Handle full algorithm result dict format
        if isinstance(solution, dict) and "solution" in solution:
            solution = solution["solution"]
        
        # Convert numpy array to dict format
        if isinstance(solution, np.ndarray):
            solution_dict = {}
            for ue_idx, bs_id in enumerate(solution):
                bs_id = int(bs_id)  # Force integer type conversion
                
                # Initialize list if key doesn't exist
                if bs_id not in solution_dict:
                    solution_dict[bs_id] = []
                
                solution_dict[bs_id].append(ue_idx)
            solution = solution_dict

        # Validate BS IDs exist
        valid_bs_ids = {int(bs.id) for bs in self.base_stations}
        for bs_id in solution.keys():
            try:
                bs_id_int = int(bs_id)
                if bs_id_int not in valid_bs_ids:
                    raise ValueError(f"Invalid BS ID {bs_id} in solution")
            except ValueError:
                raise ValueError(f"BS ID {bs_id} is not an integer")
            
        # Optional: Validate that each user index is within range.
        num_ues = len(self.ues)
        for bs_id, ue_ids in solution.items():
            for ue_id in ue_ids:
                if ue_id < 0 or ue_id >= num_ues:
                    raise IndexError(f"UE index {ue_id} for BS ID {bs_id} is out of valid range (0 to {num_ues-1}).")
        
        # Clear existing associations
        for bs in self.base_stations:
            bs.allocated_resources = {}
            self.associations[bs.id] = []

        # Apply new associations
        for bs_id, ue_ids in solution.items():
            bs_id_int = int(bs_id)
            # Find the base station matching the ID.
            bs = next(bs for bs in self.base_stations if int(bs.id) == bs_id_int)
            for ue_id in ue_ids:
                # Associate the UE with the base station.
                self.ues[ue_id].associated_bs = bs_id_int
                bs.allocated_resources[ue_id] = self.ues[ue_id].demand

        self._update_system_metrics()
    
    # def apply_solution(self, solution):
    #     """Apply a solution (either numpy array or dict) to the environment"""
    #     # Convert numpy array to dict format if necessary
    #     if isinstance(solution, np.ndarray):
    #         # Format: {bs_id: [ue_indices]}
    #         solution_dict = {}
    #         for ue_idx, bs_id in enumerate(solution):
    #             # Convert bs_id to int
    #             bs_id = int(bs_id)
    #             if bs_id not in solution_dict:
    #                 solution_dict[bs_id] = []
    #             solution_dict[bs_id].append(ue_idx)
    #         solution = solution_dict

    #     # Optionally, validate BS IDs exist (after type conversion)
    #     valid_bs_ids = {int(bs.id) for bs in self.base_stations}
    #     for bs_id in solution.keys():
    #         if int(bs_id) not in valid_bs_ids:
    #             raise ValueError(f"Invalid BS ID {bs_id} in solution")

    #     # Clear existing associations
    #     for bs in self.base_stations:
    #         bs.allocated_resources = {}
    #         self.associations[bs.id] = []

    #     # Apply new associations: convert bs.id in lookup to int as well
    #     for bs_id, ue_ids in solution.items():
    #         bs = next(bs for bs in self.base_stations if int(bs.id) == int(bs_id))
    #         for ue_id in ue_ids:
    #             self.ues[ue_id].associated_bs = bs_id
    #             bs.allocated_resources[ue_id] = self.ues[ue_id].demand

    #     self._update_system_metrics()   
    
    
    # Added evaluate_detailed_solution function
    def evaluate_detailed_solution(self, solution, alpha=0.1, beta=0.1):
        original_state = self.get_state_snapshot()
        self.apply_solution(solution)  # Ensure state is updated
                
        # rewards = [self.calculate_reward() for _ in range(10)]  # Simulate over 10 steps
        sinr_list = [ue.sinr for ue in self.ues]
        throughput_list = [torch.log2(1 + 10**(ue.sinr/10)).item() for ue in self.ues]
        bs_loads = [bs.load for bs in self.base_stations]
        
        # fitness_value = np.sum(rewards)
        fitness_value = self.calculate_reward()  # Single-step reward
        average_sinr = np.mean(sinr_list)
        average_throughput = np.mean(throughput_list)
        fairness = (np.sum(throughput_list) ** 2) / (len(throughput_list) * np.sum(np.square(throughput_list)) + 1e-6)
        load_variance = np.var(bs_loads)
        
        # Restore original state AFTER calculations
        self.set_state_snapshot(original_state)
        
        return {
            "fitness": fitness_value,
            "average_sinr": average_sinr,
            "average_throughput": average_throughput,
            "fairness": fairness,
            "load_variance": load_variance,
            "bs_loads": bs_loads
        }
        
# env = NetworkEnvironment({"num_ue": 3, "num_bs": 2})
# obs, _ = env.reset()
# print(obs["ue_0"].shape)  # Should be (2*2 + 1)=5

# actions = {"ue_0": 1, "ue_1": 0, "ue_2": 1}  # Each UE selects a BS index
# next_obs, rewards, dones, _ = env.step(actions)
# print(next_obs, rewards, dones, _ )