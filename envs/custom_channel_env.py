import sys
import os
# Add project root to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

import torch
import numpy as np
from typing import Dict, List
from hybrid_trainer.kpi_logger import KPITracker  # Import the KPI logger
class UE:
    def __init__(self, id, position, velocity, demand):
        self.id = id
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
        self.id = id
        self.position = torch.tensor(position, dtype=torch.float32)
        self.frequency = torch.tensor(frequency, dtype=torch.float32)
        self.bandwidth = torch.tensor(bandwidth , dtype=torch.float32) # MHz
        self.allocated_resources = {}  # {ue_id: allocated_bandwidth}
        self.load = 0.0
        self.capacity = bandwidth // 200  # 200 Mbps/UE

    def calculate_load(self):
        self.load = sum(self.allocated_resources.values()) / self.bandwidth

class NetworkEnvironment:
    def __init__(self, num_bs=3, num_ue=10, episode_length=100, log_kpis=True):
        self.num_bs = num_bs
        self.num_ue = num_ue
        self.episode_length = episode_length
        self.current_step = 0
        self.log_kpis = log_kpis        
        self.metaheuristic_agents = []  # Initialize empty list
        
        # Initialize KPI logger if logging is enabled
        self.kpi_logger = KPITracker() if log_kpis else None

        self.base_stations = [
            BaseStation(id=i, position=[np.random.uniform(0, 100), np.random.uniform(0, 100)],
                        frequency=100.0, bandwidth=1000.0)
            for i in range(num_bs)
        ]
        self.ues = [
            UE(id=i, position=[np.random.uniform(0, 100), np.random.uniform(0, 100)],
               velocity=[np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
               demand=np.random.randint(50, 200))
            for i in range(num_ue)
        ]
        self.associations = {bs.id: [] for bs in self.base_stations}
        
    def reset(self):
        self.current_step = 0
        for ue in self.ues:
            ue.position = torch.tensor([np.random.uniform(0, 100), np.random.uniform(0, 100)])
            ue.associated_bs = None
        return self._get_obs()

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

    def step(self, actions: Dict[str, int]):
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
        
        if self.log_kpis:
            self.kpi_logger.log(self.current_step, self.ues, self.base_stations)
        
        return self._get_obs(), reward, done, {}
    
    def _get_obs(self):
        #return {f"BS_{bs.id}": {"load": bs.load} for bs in self.base_stations}
        return {  # Explicitly cast to float32
            f"BS_{bs.id}": {"load": np.float32(bs.load)} for bs in self.base_stations
            }
    
    # In NetworkEnvironment.get_current_state() for visualization
    def get_current_state(self):
        return {
            "base_stations": [
                {
                    "id": bs.id,
                    "position": bs.position.numpy(),
                    "load": bs.load,
                    "capacity": bs.capacity
                } for bs in self.base_stations
            ],
            "users": [
                {
                    "id": ue.id,
                    "position": ue.position.numpy(),
                    "associated_bs": ue.associated_bs,
                    "sinr": ue.sinr.item()  # Convert tensor to float
                } for ue in self.ues
            ],
            "associations": self.associations.copy(),
            "metaheuristic_agents": self.current_metaheuristic_agents  # Set by optimizer
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
        """Apply metaheuristic solution with validation"""
        # Clear existing associations
        for bs in self.base_stations:
            bs.allocated_resources = {}
            self.associations[bs.id] = []
        # Validate BS IDs exist
        valid_bs_ids = {bs.id for bs in self.base_stations}
        for bs_id in solution.keys():
            if bs_id not in valid_bs_ids:
                raise ValueError(f"Invalid BS ID {bs_id} in solution")
        
        # # Apply associations
        # for bs_id, ue_ids in solution.items():
        #     bs = next(bs for bs in self.base_stations if bs.id == bs_id)
        #     bs.allocated_resources = {}
        #     for ue_id in ue_ids:
        #         self.ues[ue_id].associated_bs = bs_id
        # Apply new associations
        for bs_id, ue_ids in solution.items():
            bs = next(bs for bs in self.base_stations if bs.id == bs_id)
            for ue_id in ue_ids:
                self.ues[ue_id].associated_bs = bs_id
                bs.allocated_resources[ue_id] = self.ues[ue_id].demand
        self._update_system_metrics()
    
    # Added evaluate_detailed_solution function
    def evaluate_detailed_solution(self, solution, alpha=0.1, beta=0.1):
        original_state = self.get_state_snapshot()
        self.apply_solution(solution)  # Ensure state is updated
                
        rewards = [self.calculate_reward() for _ in range(10)]  # Simulate over 10 steps
        sinr_list = [ue.sinr for ue in self.ues]
        throughput_list = [torch.log2(1 + 10**(ue.sinr/10)).item() for ue in self.ues]
        bs_loads = [bs.load for bs in self.base_stations]
        
        fitness_value = np.sum(rewards)
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
    