import sys
import os
# Add project root to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

import torch
import numpy as np
import gymnasium as gym
from ray.rllib.env import EnvContext
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

class NetworkEnvironment(gym.Env):
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
        
        
        

        self.base_stations = [
            BaseStation(id=i, position=[np.random.uniform(0, 100), np.random.uniform(0, 100)],
                        frequency=100.0, bandwidth=1000.0)
            for i in range(self.num_bs)
        ]
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
        # Observation space for each BS agent (4 features + 2*num_ue positions)
        self.observation_space = gym.spaces.Dict({
            f"bs_{i}": gym.spaces.Box(
                low=-np.inf, 
                high=np.inf,
                shape=(3 + 2*self.num_ue,), 
                dtype=np.float32
            ) for i in range(self.num_bs)
        })
                # Define action space with proper bounds
        # Action space for each BS agent (UE selection mask)
        self.action_space = gym.spaces.Dict({
            f"bs_{i}": gym.spaces.MultiBinary(self.num_ue)
            for i in range(self.num_bs)
        })
    # def reset(self):
    #     self.current_step = 0
    #     self.version += 1  # Increment on state change
    #     for ue in self.ues:
    #         ue.position = torch.tensor([np.random.uniform(0, 100), np.random.uniform(0, 100)])
    #         ue.associated_bs = None
    #     return self._get_obs()
    
    def reset(self, seed=None, options=None):
            # Called automatically by RLlib at episode start
        self.current_step = 0
        self.version += 1
            
            # Reset UE positions and associations
        for ue in self.ues:
            ue.position = torch.tensor(
                    [np.random.uniform(0, 100), np.random.uniform(0, 100)],
                    dtype=torch.float32  # Match observation dtype
                )
            ue.associated_bs = None
            
            # Generate initial observations
            obs = self._get_obs()
            
            # Convert all values to float32 explicitly
        return {
                agent_id: obs[agent_id].astype(np.float32)
                for agent_id in obs
            }, {}

    def calculate_sinr(self, ue, bs):
        distance = torch.norm(ue.position - bs.position)
        path_loss = 32.4 + 20 * torch.log10(distance + 1e-6) + 20 * torch.log10(bs.frequency)
        tx_power = 30  # dBm
        noise_floor = -174 + 10 * torch.log10(bs.bandwidth * 1e6)
        sinr_linear = 10 ** ((tx_power - path_loss - noise_floor) / 10)
        return 10 * torch.log10(sinr_linear + 1e-10)

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
    
    # def step(self, actions: Dict[str, int]):
    #     self.version += 1
    #     for ue in self.ues:
    #         ue.update_position()  # Update UE positions

    #     # Process actions to associate UEs with BSs
    #     for bs_id, ue_ids in actions.items():
    #         bs = next(bs for bs in self.base_stations if bs.id == int(bs_id.split("_")[1]))
    #         for ue_id in ue_ids:
    #             ue = self.ues[ue_id]
    #             ue.associated_bs = bs.id
    #             # Calculate allocated resources proportionally
    #             total_demand = sum(ue.demand for ue in self.ues if ue.associated_bs == bs.id)
    #             bs.allocated_resources[ue_id] = ue.demand / total_demand if total_demand != 0 else 0.0

    #     # Calculate BS load and UE SINR
    #     for bs in self.base_stations:
    #         bs.calculate_load()
        
    #     sinr_values = []
    #     for ue in self.ues:
    #         if ue.associated_bs is not None:
    #             bs = next(bs for bs in self.base_stations if bs.id == ue.associated_bs)
    #             ue.sinr = self.calculate_sinr(ue, bs)
    #             sinr_values.append(ue.sinr)
    #         else:
    #             sinr_values.append(0.0)  # Handle unassociated UEs

    #     # Calculate reward and metrics
    #     reward = self.calculate_reward()
    #     avg_sinr = np.mean(sinr_values) if sinr_values else 0.0
    #     fairness = (sum(sinr_values) ** 2) / (len(sinr_values) * sum(s**2 for s in sinr_values)) if sinr_values else 0.0
    #     load_variance = np.var([bs.load for bs in self.base_stations])

    #     # Populate info dict with metrics
    #     info = {
    #         "custom_metrics": {
    #             "sinr_mean": avg_sinr,
    #             "fairness": fairness,
    #             "load_variance": load_variance
    #         }
    #     }

    #     self.current_step += 1
    #     done = self.current_step >= self.episode_length

    #     return self._get_obs(), reward, done, info
    def step(self, actions: Dict[str, int]):
        self.version += 1  # Increment on state change
        for ue in self.ues:
            ue.update_position()  # Actually move UEs each step
            
        for bs_id, ue_ids in actions.items():
            bs = next(bs for bs in self.base_stations if bs.id == int(bs_id.split("_")[1]))
            for ue_id in ue_ids:
                ue = self.ues[ue_id]
                ue.associated_bs = bs.id
                bs.allocated_resources[ue_id] = ue.demand / sum(ue.demand for ue in self.ues if ue.associated_bs == bs.id)
        
        for bs in self.base_stations:
            bs.calculate_load()
        for ue in self.ues:
            if ue.associated_bs is not None:
                bs = next(bs for bs in self.base_stations if bs.id == ue.associated_bs)
                ue.sinr = self.calculate_sinr(ue, bs)
        
        reward = self.calculate_reward()
        self.current_step += 1
        done = self.current_step >= self.episode_length
        
        # if self.log_kpis:
        #     self.kpi_logger.log(self.current_step, self.ues, self.base_stations)
        
        return self._get_obs(), reward, done, {}
    
    # def _get_obs(self):
        
    #     return {  # Explicitly cast to float32
    #         f"BS_{bs.id}": {"load": np.float32(bs.load)} for bs in self.base_stations
    #         }
    
    def _get_obs(self):
        """Structure observations per agent with dimension consistency"""
        return {
            f"bs_{bs.id}": np.concatenate([
                np.array([bs.load], dtype=np.float32).flatten(),  # 1D array
                bs.position.numpy().flatten().astype(np.float32),  # 1D array
                np.concatenate([ue.position.numpy() for ue in self.ues])  # Flattened 1D
            ])
            for bs in self.base_stations
        }
        

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
    