import sys
import os

# Add project root to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

import torch
import numpy as np
from typing import Dict, List
from kpi_logger import KPI_Logger  # Import the KPI logger

class UE:
    def __init__(self, id, position, velocity, demand):
        self.id = id
        self.position = torch.tensor(position, dtype=torch.float32)
        self.velocity = torch.tensor(velocity, dtype=torch.float32)
        self.demand = demand  # Mbps
        self.associated_bs = None
        self.sinr = 0.0

    def update_position(self, delta_time=1.0):
        self.position += self.velocity * delta_time + torch.randn(2) * 0.1

class BaseStation:
    def __init__(self, id, position, frequency, bandwidth):
        self.id = id
        self.position = torch.tensor(position, dtype=torch.float32)
        self.frequency = frequency  # GHz
        self.bandwidth = bandwidth  # MHz
        self.allocated_resources = {}  # {ue_id: allocated_bandwidth}
        self.load = 0.0

    def calculate_load(self):
        self.load = sum(self.allocated_resources.values()) / self.bandwidth

class NetworkEnvironment:
    def __init__(self, num_bs=3, num_ue=10, episode_length=100, log_kpis=True):
        self.num_bs = num_bs
        self.num_ue = num_ue
        self.episode_length = episode_length
        self.current_step = 0
        self.log_kpis = log_kpis
        
        # Initialize KPI logger if logging is enabled
        self.kpi_logger = KPI_Logger() if log_kpis else None

        self.base_stations = [
            BaseStation(id=i, position=[np.random.uniform(0,100), np.random.uniform(0,100)],
                        frequency=100, bandwidth=1000)
            for i in range(num_bs)
        ]
        self.ues = [
            UE(id=i, position=[np.random.uniform(0,100), np.random.uniform(0,100)],
               velocity=[np.random.uniform(-1,1), np.random.uniform(-1,1)],
               demand=np.random.randint(50, 200))
            for i in range(num_ue)
        ]

    def reset(self):
        self.current_step = 0
        for ue in self.ues:
            ue.position = torch.tensor([np.random.uniform(0,100), np.random.uniform(0,100)])
            ue.associated_bs = None
        return self._get_obs()

    def _calculate_sinr(self, ue, bs):
        distance = torch.norm(ue.position - bs.position)
        path_loss = 32.4 + 20 * torch.log10(distance + 1e-6) + 20 * torch.log10(bs.frequency)
        tx_power = 30
        noise_floor = -174 + 10 * torch.log10(bs.bandwidth * 1e6)
        sinr_linear = 10 ** ((tx_power - path_loss - noise_floor) / 10)
        return 10 * torch.log10(sinr_linear + 1e-10)

    def _calculate_reward(self):
        throughput = sum(torch.log2(1 + 10**(ue.sinr/10)) for ue in self.ues)
        loads = torch.tensor([bs.load for bs in self.base_stations])
        jain = (loads.sum() ** 2) / (self.num_bs * (loads ** 2).sum())
        overload_penalty = torch.sum(torch.relu(loads - 1.0))
        return throughput + 2.0 * jain - 0.5 * overload_penalty

    def step(self, actions: Dict[str, int]):
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
                ue.sinr = self._calculate_sinr(ue, bs)
        
        reward = self._calculate_reward()
        self.current_step += 1
        done = self.current_step >= self.episode_length
        
        if self.log_kpis:
            self.kpi_logger.log(self.current_step, self.ues, self.base_stations)
        
        return self._get_obs(), reward, done, {}
    
    def _get_obs(self):
        return {f"BS_{bs.id}": {"load": bs.load} for bs in self.base_stations}
