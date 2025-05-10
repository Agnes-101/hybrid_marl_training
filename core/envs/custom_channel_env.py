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
import time
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
        self.capacity = bandwidth *0.8  # 800 Mbps/UE

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
        
        self.ues = [
            UE(id=i, position=[np.random.uniform(0, 100), np.random.uniform(0, 100)],
            velocity=[np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
            demand=np.random.randint(50, 200))
            for i in range(self.num_ue)
        ]
        print(f"Created {len(self.ues)} UEs")
        self.associations = {bs.id: [] for bs in self.base_stations}
        # Initialize KPI logger if logging is enabled
        self.kpi_logger = KPITracker() if log_kpis else None        
        self.observation_space = gym.spaces.Dict({
            f"ue_{i}": gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(3 * self.num_bs + 1,)  # SINRs + BS loads + BS Utilizations + own demand
                # dtype=np.float32
            ) for i in range(self.num_ue)
        })

        # Action space: UE chooses a BS (Discrete(num_bs))
        self.action_space = gym.spaces.Dict({
            f"ue_{i}": gym.spaces.Discrete(self.num_bs) 
            for i in range(self.num_ue)
        })
        
    def observation_space(self, agent_id=None):
        if agent_id is not None:
            return self.observation_space[agent_id]
        return self.observation_space

    def action_space(self, agent_id=None):
        if agent_id is not None:
            return self.action_space[agent_id]
        return self.action_space

    def reward(self, agent):
        return self.calculate_individual_reward(agent)  # Implement per-BS reward

    def calculate_individual_reward(self, agent_id=None):
        if agent_id is None:
            return 0.0
            
        # Extract UE ID from agent ID
        if isinstance(agent_id, str) and agent_id.startswith("ue_"):
            ue_id = int(agent_id.split("_")[1])
            ue = self.ues[ue_id]
            
            if ue.associated_bs is None:
                return -10.0  # Penalize unconnected UEs, but not too extreme
            
            bs = self.base_stations[ue.associated_bs]
            sinr_factor = max(0, min(1, (ue.sinr + 10) / 40))  # Normalize to [0,1]
            
            # Reward balancing load and maximizing SINR
            load_factor = 1 - (bs.load / bs.capacity)
            reward = (0.7 * sinr_factor + 0.3 * load_factor) * 10.0
            
            return reward
        
        return 0.0
    
    # def reset(self, seed=None, options=None):
    #     self.current_step = 0
    #     # Reset UE positions/associations
    #     for ue in self.ues:
    #         ue.position = np.random.uniform(0, 100, size=2)  # Use numpy, not PyTorch
    #         ue.associated_bs = None
                
    #     # Return observations for ALL UE agents
    #     return self._get_obs(), {}  # Return observation and info dict
    
    def reset(self, seed=None, options=None):
        self.current_step = 0
        # Reset UE positions/associations
        for ue in self.ues:
            ue.position = np.random.uniform(0, 100, size=2)
            ue.associated_bs = None
            ue.sinr = -np.inf  # Initialize SINR
                    
        # Return observations for ALL UE agents
        return self._get_obs(), {}

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
        
    # def step(self, actions: Dict[str, int]):
    #     # 1. Process UE actions (BS selections)
    #     for agent_id, bs_choice in actions.items():
    #         ue_id = int(agent_id.split("_")[1])
    #         ue = self.ues[ue_id]
    #         target_bs = self.base_stations[bs_choice]
            
    #         # Disassociate from old BS (if any)
    #         if ue.associated_bs is not None:
    #             old_bs = self.base_stations[ue.associated_bs]
    #             if ue.id in old_bs.allocated_resources:
    #                 del old_bs.allocated_resources[ue.id]
            
    #         # Associate with new BS if capacity allows
    #         if target_bs.load + ue.demand <= target_bs.capacity:
    #             target_bs.allocated_resources[ue.id] = ue.demand
    #             ue.associated_bs = bs_choice
        
    #     # 2. Update BS loads and UE SINRs
    #     for bs in self.base_stations:
    #         bs.load = sum(bs.allocated_resources.values())
        
    #     # 3. Calculate rewards (individual + global)
    #     rewards = {}
    #     # global_fairness = self.calculate_jains_fairness()  # From earlier code
    #     for ue in self.ues:
    #         if ue.associated_bs is None:
    #             reward = -1.0  # Penalize unconnected UEs
    #         else:
    #             bs = self.base_stations[ue.associated_bs]
    #             throughput = np.log2(1 + 10**(ue.sinr/10))
    #             # reward = throughput + 0.5 * global_fairness  # Hybrid reward
    #             reward = self.calculate_individual_reward()
            
    #         rewards[f"ue_{ue.id}"] = reward
        
    #     # 4. Check termination
    #     self.current_step += 1
    #     done = {"__all__": self.current_step >= self.episode_length}
        
    #     return self._get_obs(), rewards, done, {}
    
    def _calculate_sinrs(self, ue):
        bs_positions = np.array([bs.position for bs in self.base_stations])
        distances = np.linalg.norm(bs_positions - ue.position, axis=1)
        distances = np.clip(distances, 1e-8, None)  # Prevent log(0)
        sinrs = np.clip(30.0 - 20 * np.log10(distances), -30, 60)  # Bound SINR values          
        # return 30.0 - 20 * np.log10(distances + 1e-6)
        return sinrs
    
    def _update_sinrs(self):
        for ue in self.ues:
            if ue.associated_bs is not None:
                sinrs = self._calculate_sinrs(ue)
                ue.sinr = sinrs[ue.associated_bs]
            else:
                ue.sinr = -np.inf
    
    
    def step(self, actions: Dict[str, int]):
        # Add timing for performance analysis
        print(f"Step called with {len(actions)} actions")
        try:
            start_time = time.time()
            
            # Pre-extract all UE and BS information for faster processing
            ue_positions = np.array([ue.position for ue in self.ues])
            bs_positions = np.array([bs.position for bs in self.base_stations])
            bs_capacities = np.array([bs.capacity for bs in self.base_stations])
            
            # Track allocations by BS
            bs_allocations = {bs.id: 0 for bs in self.base_stations}
            bs_capacity_used = {bs.id: 0 for bs in self.base_stations}
            
            # Sort UEs by demand (smallest first) for better allocation
            ue_actions = [(int(agent_id.split("_")[1]), bs_choice) 
                        for agent_id, bs_choice in actions.items()]
            ue_actions.sort(key=lambda x: self.ues[x[0]].demand)
            
            # Reset all allocations
            for bs in self.base_stations:
                bs.allocated_resources = {}
                bs.load = 0.0
            
            connected_count = 0
            
            # Process actions in sorted order - use numpy operations where possible
            for ue_id, bs_choice in ue_actions:
                ue = self.ues[ue_id]
                target_bs = self.base_stations[bs_choice]
                

                # Associate with BS if capacity allows
                if target_bs.load + ue.demand <= target_bs.capacity:
                    target_bs.allocated_resources[ue.id] = ue.demand
                    target_bs.load += ue.demand
                    ue.associated_bs = bs_choice
                    bs_allocations[bs_choice] += 1
                    bs_capacity_used[bs_choice] += ue.demand
                    connected_count += 1
                else:
                    # Calculate SINRs for this UE to all BSs
                    distances = np.linalg.norm(bs_positions - ue.position, axis=1)
                    sinrs = 30.0 - 20 * np.log10(distances + 1e-6)  # Simplified model
                    
                    # Sort BSs by SINR (descending)
                    sorted_bs_indices = np.argsort(-sinrs)
                    
                    # Try finding alternative BS
                    for alt_bs_id in sorted_bs_indices:
                        alt_bs = self.base_stations[alt_bs_id]
                        if alt_bs.load + ue.demand <= alt_bs.capacity:
                            alt_bs.allocated_resources[ue.id] = ue.demand
                            alt_bs.load += ue.demand
                            ue.associated_bs = alt_bs_id
                            bs_allocations[alt_bs_id] += 1
                            bs_capacity_used[alt_bs_id] += ue.demand
                            connected_count += 1
                            # chosen_id = ue.associated_bs
                            # print(f"UE{ue_id}: chosen BS={chosen_id}, SINR={sinrs[chosen_id]:.2f}")
                            break
                    else:
                        ue.associated_bs = None
            
            # Update SINR values for all UEs in one pass
            for ue in self.ues:
                if ue.associated_bs is not None:
                    # Calculate SINR for this UE
                    distance = np.linalg.norm(bs_positions[ue.associated_bs] - ue.position)
                    ue.sinr = 30.0 - 20 * np.log10(distance + 1e-6)  # Simplified model
                else:
                    ue.sinr = -np.inf
            
            # Calculate rewards efficiently
            rewards = {}
            total_reward = 0
            for bs in self.base_stations:
                print(f"BS {bs.id}: load={bs.load}, capacity={bs.capacity}, ratio={bs.load/bs.capacity if bs.capacity>0 else 'inf'}")
            for ue in self.ues:                               
                if ue.associated_bs is None:
                    reward = -1.0  # Penalize unconnected UEs
                else:
                    bs = self.base_stations[ue.associated_bs]
                    throughput = np.log2(1 + 10**(ue.sinr/10))
                    
                    # Add stronger incentive for load balancing
                    load_factor = 1 - (bs.load / bs.capacity)
                    
                    # Higher reward for connecting to less loaded BS
                    reward = throughput * load_factor * 5.0 
                    # Inside your reward calculation:
                    
                reward_scale_factor = 1.0 / np.sqrt(self.num_ue)
                # rewards[f"ue_{ue.id}"] = reward * reward_scale_factor
                clipped =np.clip(reward * reward_scale_factor, -10.0, 10.0)
                rewards[f"ue_{ue.id}"] = clipped
                total_reward += reward
                # if ue_id % 10 == 0:  # Only print every 10th UE to avoid excessive output
                #         print(f"UE {ue_id}: connected={ue.associated_bs is not None}, "
                #             f"sinr={ue.sinr:.2f}, throughput={throughput:.2f}, "
                #             f"load_factor={load_factor:.2f}, reward={rewards[f'ue_{ue.id}']:.4f}")
                        
            print(f"Connected Users : {connected_count} Users")
            # Log performance metrics at regular intervals
            step_time = time.time() - start_time
            if self.current_step % 10 == 0 or connected_count < self.num_ue:
                print(f"Step {self.current_step} | Time: {step_time:.3f}s | Connected: {connected_count}/{self.num_ue} UEs")
                
                # Calculate load balancing metrics
                loads = [bs.load for bs in self.base_stations]
                capacities = [bs.capacity for bs in self.base_stations]
                utilizations = [load/cap if cap > 0 else 0 for load, cap in zip(loads, capacities)]
                
                # Jain's fairness index for load distribution
                if sum(utilizations) > 0:
                    squared_sum = sum(utilizations)**2
                    sum_squared = sum(u**2 for u in utilizations) * len(utilizations)
                    jains_index = squared_sum / sum_squared if sum_squared > 0 else 0
                    print(f"Load Balancing Fairness: {jains_index:.4f}")
                
                # Log detailed BS allocations if not all UEs connected
                if connected_count < self.num_ue:
                    print(f"BS allocations: {bs_allocations}")
                    print(f"BS capacity %: {[bs_capacity_used[bs.id]/bs.capacity if bs.capacity > 0 else 0 for bs in self.base_stations]}")
            
            
            # Increment step counter
            self.current_step += 1
            # Track current solution for visualization
            current_solution = []
            for ue in self.ues:
                if ue.associated_bs is not None:
                    current_solution.append(ue.associated_bs)
                else:
                    # Use a default value for unconnected UEs
                    current_solution.append(0)  # Or None if your visualization can handle it
            print(f"Current solution at : {current_solution}")
            sinr_list = [ue.sinr if ue.associated_bs is not None else -np.inf
             for ue in self.ues]
            
            if self.log_kpis and hasattr(self, 'kpi_logger') and self.kpi_logger is not None:
                # Performance metrics
                metrics = {
                    "connected_ratio": connected_count / self.num_ue,
                    "step_time": step_time,
                    "episode_reward_mean": total_reward / self.num_ue if self.num_ue > 0 else 0,
                    "fairness_index": jains_index if 'jains_index' in locals() else 0,
                    "throughput": sum([np.log2(1 + 10**(ue.sinr/10)) for ue in self.ues if ue.associated_bs is not None]),
                    "solution":       current_solution,
                    "sinr_list":      sinr_list,
                    # Add more metrics as needed
                }
                
                # Log at the environment level instead of through callback
                self.kpi_logger.log_metrics(
                    phase="environment", 
                    algorithm="hybrid_marl",
                    metrics=metrics,
                    episode=self.current_step-1
                )
                
            # Split termination into terminated and truncated (for Gymnasium compatibility)
            terminated = {"__all__": False}  # Episode is not terminated due to failure condition
            truncated = {"__all__": self.current_step >= self.episode_length}  # Episode length limit reached    
            # Common info for all agents
            common_info = {
                "connected_ratio": connected_count / self.num_ue,
                "step_time": step_time,
                "current solution":current_solution,
                "avg_reward": total_reward / self.num_ue if self.num_ue > 0 else 0
            }
            
            # Create info dict with one entry per agent, plus common info
            info = {
                f"ue_{ue.id}": {
                    "connected": ue.associated_bs is not None,
                    "sinr": float(ue.sinr)
                } for ue in self.ues  # Add minimal agent-specific info
            }
            info["__common__"] = common_info  # Add common info under special key
            # Save as last info
            self.last_info = info
            return self._get_obs(), rewards, terminated, truncated, info
        
        except Exception as e:
            print(f"ERROR in step: {e}")
            import traceback
            print(traceback.format_exc())
            # Return a safe default response
            return self._get_obs(), {f"ue_{ue.id}": 0.0 for ue in self.ues}, {"__all__": False}, {"__all__": True}, {"__common__": {"error": str(e)}}
        
    # Add this to your NetworkEnvironment class
    def get_last_info(self):
        """Return the last info dict from a step"""
        if hasattr(self, 'last_info'):
            print("Getting lastest info....")
            return self.last_info
        return None
    # def _get_obs(self):
    #     # Precompute BS loads and positions once
    #     bs_loads = np.array([bs.load for bs in self.base_stations], dtype=np.float32)
    #     bs_positions = np.array([bs.position for bs in self.base_stations], dtype=np.float32)
                
    #     obs = {}
    #     for ue in self.ues:
    #         # Calculate SINR to all BSs (vectorized)
    #         distances = np.linalg.norm(bs_positions - ue.position, axis=1)
    #         sinrs = 30.0 - 20 * np.log10(distances + 1e-6)  # Simplified model
            
    #         # Store SINR for the associated BS (if any)
    #         if ue.associated_bs is not None:
    #             ue.sinr = sinrs[ue.associated_bs]  # Add this line
    #         else:
    #             ue.sinr = -np.inf  # No connection
            
    #         # Normalize features
    #         normalized_sinrs = sinrs / 40.0  # Assume max SINR ~40 dB
    #         normalized_loads = bs_loads / 1000.0  # Assuming max load=1000
                        
    #         obs[f"ue_{ue.id}"] = np.concatenate([
    #             normalized_sinrs,
    #             normalized_loads,
    #             [ue.demand / 200.0]  # Max demand=200
    #         ], dtype=np.float32)
            
            
    #     return obs
    
    def _get_obs(self):
        # Precompute BS loads and positions once
        bs_loads = np.array([bs.load for bs in self.base_stations], dtype=np.float32)
        bs_positions = np.array([bs.position for bs in self.base_stations], dtype=np.float32)
        
        # Create normalized_loads and normalized_sinrs as arrays just once
        normalized_loads = bs_loads / 1000.0  # Assuming max load=1000
        
        # Calculate utilizations once
        bs_utilizations = np.array([min(bs.load / bs.capacity, 1.0) if bs.capacity > 0 else 0.0 
                                for bs in self.base_stations], dtype=np.float32)
                
        obs = {}
        for ue in self.ues:
            # Calculate SINR to all BSs (vectorized)
            distances = np.maximum(np.linalg.norm(bs_positions - ue.position, axis=1), 1e-6)
            sinrs = np.clip(30.0 - 20 * np.log10(distances), -30, 60)  # Bounded SINR
            
            # Store SINR for the associated BS (if any)
            if ue.associated_bs is not None:
                ue.sinr = sinrs[ue.associated_bs]  # Add this line
            else:
                ue.sinr = -np.inf  # No connection
            
            # Normalize features
            normalized_sinrs = sinrs / 40.0  # Assume max SINR ~40 dB
                        
            obs[f"ue_{ue.id}"] = np.concatenate([
                normalized_sinrs,
                normalized_loads,
                bs_utilizations,  # Add BS utilization as suggested
                [ue.demand / 200.0]  # Max demand=200
            ], dtype=np.float32)
                
        return obs

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
        
# import sys
# import os
# # Add project root to Python's path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, project_root) if project_root not in sys.path else None

# import torch
# import numpy as np
# import gymnasium as gym
# from ray.rllib.env import EnvContext
# from ray.rllib.env.multi_agent_env import MultiAgentEnv
# from typing import Dict, List
# from hybrid_trainer.kpi_logger import KPITracker  # Import the KPI logger
# import time
# class UE:
#     def __init__(self, id, position, velocity, demand, ewma_alpha=0.9):
#         self.id = int(id)
#         self.position = np.array(position, dtype=np.float32)
#         self.velocity = np.array(velocity, dtype=np.float32)
#         self.demand = float(demand)  # Mbps
#         self.associated_bs = None
#         self.sinr = -np.inf
#         # EWMA of data rate for proportional-fair
#         self.ewma_dr = 0.0
#         self.ewma_alpha = ewma_alpha

#     def update_position(self, delta_time=1.0):
#         noise = np.random.randn(2) * 0.1
#         self.position = self.position + self.velocity * delta_time + noise

#     def update_ewma(self, measured_dr):
#         # EWMA update for historical average data rate
#         self.ewma_dr = self.ewma_alpha * self.ewma_dr + (1 - self.ewma_alpha) * measured_dr

# class BaseStation:
#     def __init__(self, id, position, frequency, bandwidth, height=50.0, tx_power_dbm=30.0):
#         self.id = int(id)
#         self.position = np.array(position, dtype=np.float32)
#         self.frequency = float(frequency)    # MHz
#         self.bandwidth = float(bandwidth)    # MHz
#         self.height = float(height)          # m
#         self.tx_power = float(tx_power_dbm)  # dBm
#         self.allocated_resources = {}        # {ue_id: allocated_rate}
#         self.load = 0.0                      # sum of allocated rates
#         self.capacity = self.bandwidth * 0.8 # e.g. 80% of bandwidth in Mbps

#     def calculate_load(self):
#         self.load = sum(self.allocated_resources.values())

#     def path_loss(self, distance, ue_height=1.5):
#         # Okumura-Hata suburban model
#         f = self.frequency
#         hb = self.height
#         hu = ue_height
#         ch = 0.8 + (1.1 * np.log10(f) - 0.7) * hu - 1.56 * np.log10(f)
#         const1 = 69.55 + 26.16 * np.log10(f) - 13.82 * np.log10(hb) - ch
#         const2 = 44.9 - 6.55 * np.log10(hb)
#         return const1 + const2 * np.log10(distance + 1e-9)

#     def noise_mW(self):
#         # Thermal noise: -174 dBm/Hz + 10*log10(BW_Hz)
#         noise_dbm = -174 + 10 * np.log10(self.bandwidth * 1e6)
#         return 10 ** (noise_dbm / 10)
    
#     def _calculate_local_interference(self, neighbor_dist=100.0):
#         """Calculate interference at this BS from all other BSs within neighbor_dist."""
#         interference = 0.0
#         for other in self.base_stations:
#             if other.id == self.id:
#                 continue
#             d = np.linalg.norm(self.position - other.position)
#             if d < neighbor_dist:
#                 # received power from interfering BS
#                 prx_int = other.received_power_mW(self.position)
#                 interference += prx_int
#         return interference
#     def received_power_mW(self, ue_pos):
#         d = np.linalg.norm(self.position - ue_pos)
#         L = self.path_loss(d)        
#         return 10 ** ((self.tx_power  - L) / 10) # + self.beamforming_gain_db

#     def snr_linear(self, ue_pos):
#         d = np.linalg.norm(self.position - ue_pos)
#         L = self.path_loss(d)
#         prx_mw = 10 ** ((self.tx_power - L) / 10)
#         return prx_mw / self.noise_mW()

#     def data_rate_unshared(self, ue):
#         snr = self.snr_linear(ue.position)
#         return self.bandwidth * np.log2(1 + snr)

#     def priority(self, ue, alpha=1.0, beta=1.0, eps=1e-6):
#         T_inst = self.data_rate_unshared(ue)
#         R_hist = ue.ewma_dr
#         return (T_inst ** alpha) / (R_hist ** beta + eps)

#     def data_rate_shared(self, ue, alpha=1.0, beta=1.0):
#         # Add ue temporarily if not present
#         if ue not in self.allocated_resources:
#             self.allocated_resources[ue.id] = 0.0
#         # Compute priorities for all connected UEs
#         ps = [self.priority(u, alpha, beta) for u_id in self.allocated_resources for u in [ue] if True]
#         sum_p = sum(ps) + 1e-9
#         # Fraction of resources for this UE
#         p_i = self.priority(ue, alpha, beta)
#         fraction = p_i / sum_p
#         T_i = self.data_rate_unshared(ue)
#         return fraction * T_i

# class NetworkEnvironment(MultiAgentEnv):
#     def __init__(self, config:EnvContext, log_kpis=True):        
#         super().__init__()  # ✅ Initialize gym.Env
#         self.config = config
#         # Define observation space first
#         # Access passed environment instance
#         self.num_bs = config.get("num_bs", 20)
#         self.num_ue = config.get("num_ue", 200)
#         self.episode_length = config.get("episode_length", 100)
#         self.env_instance = config.get("environment_instance")
        
#         self.version = 0  # Internal state version
#         self.current_step = 0
#         self.log_kpis = log_kpis        
#         self.metaheuristic_agents = []  # Initialize empty list         
#         # Calculate grid dimensions for base stations
#         num_bs = self.num_bs
#         a = int(np.floor(np.sqrt(num_bs)))
#         rows, cols = 1, num_bs  # Default to 1 row if no factors found
#         while a > 0:
#             if num_bs % a == 0:
#                 rows = a
#                 cols = num_bs // a
#                 break
#             a -= 1
        
#         # Calculate spacing to avoid edges
#         x_spacing = 100.0 / (cols + 1)
#         y_spacing = 100.0 / (rows + 1)
        
#         # Generate grid positions
#         positions = []
#         for i in range(rows):
#             for j in range(cols):
#                 x = (j + 1) * x_spacing
#                 y = (i + 1) * y_spacing
#                 positions.append([x, y])
#                 if len(positions) == num_bs:
#                     break  # Exit early if we've reached the desired number
#             if len(positions) == num_bs:
#                 break
        
#         self.base_stations = [
#             BaseStation(id=i, position=positions[i],
#                         frequency=100.0, bandwidth=1000.0)
#             for i in range(self.num_bs)
#         ]
        
#         self.ues = [
#             UE(id=i, position=[np.random.uniform(0, 100), np.random.uniform(0, 100)],
#             velocity=[np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
#             demand=np.random.randint(50, 200))
#             for i in range(self.num_ue)
#         ]
#         print(f"Created {len(self.ues)} UEs")
#         self.associations = {bs.id: [] for bs in self.base_stations}
#         # Initialize KPI logger if logging is enabled
#         self.kpi_logger = KPITracker() if log_kpis else None        
#         self.observation_space = gym.spaces.Dict({
#             f"ue_{i}": gym.spaces.Box(
#                 low=-np.inf, 
#                 high=np.inf, 
#                 shape=(3 * self.num_bs + 1,)  # SINRs + BS loads + BS Utilizations + own demand
#                 # dtype=np.float32
#             ) for i in range(self.num_ue)
#         })

#         # Action space: UE chooses a BS (Discrete(num_bs))
#         self.action_space = gym.spaces.Dict({
#             f"ue_{i}": gym.spaces.Discrete(self.num_bs) 
#             for i in range(self.num_ue)
#         })
        
#     def observation_space(self, agent_id=None):
#         if agent_id is not None:
#             return self.observation_space[agent_id]
#         return self.observation_space

#     def action_space(self, agent_id=None):
#         if agent_id is not None:
#             return self.action_space[agent_id]
#         return self.action_space

#     def reward(self, agent):
#         return self.calculate_individual_reward(agent)  # Implement per-BS reward

#     def calculate_individual_reward(self, agent_id=None):
#         if agent_id is None:
#             return 0.0
            
#         # Extract UE ID from agent ID
#         if isinstance(agent_id, str) and agent_id.startswith("ue_"):
#             ue_id = int(agent_id.split("_")[1])
#             ue = self.ues[ue_id]
            
#             if ue.associated_bs is None:
#                 return -10.0  # Penalize unconnected UEs, but not too extreme
            
#             bs = self.base_stations[ue.associated_bs]
#             sinr_factor = max(0, min(1, (ue.sinr + 10) / 40))  # Normalize to [0,1]
            
#             # Reward balancing load and maximizing SINR
#             load_factor = 1 - (bs.load / bs.capacity)
#             reward = (0.7 * sinr_factor + 0.3 * load_factor) * 10.0
            
#             return reward
        
#         return 0.0
    
#     # def reset(self, seed=None, options=None):
#     #     self.current_step = 0
#     #     # Reset UE positions/associations
#     #     for ue in self.ues:
#     #         ue.position = np.random.uniform(0, 100, size=2)  # Use numpy, not PyTorch
#     #         ue.associated_bs = None
                
#     #     # Return observations for ALL UE agents
#     #     return self._get_obs(), {}  # Return observation and info dict
    
#     def pick_best_bs(self, ue, w1=0.7, w2=0.3):
#         unshared = np.array([bs.data_rate_unshared(ue) for bs in self.base_stations])
#         max_unsh = unshared.max() + 1e-9
#         loads = np.array([bs.load for bs in self.base_stations])
#         caps  = np.array([bs.capacity for bs in self.base_stations])
#         load_factor = 1 - loads / caps
#         scores = w1 * (unshared / max_unsh) + w2 * load_factor
#         return int(np.argmax(scores))

#     def reset(self, seed=None, options=None):
#         self.current_step = 0
#         for ue in self.ues:
#             ue.position = np.random.uniform(0,100,2).astype(np.float32)
#             ue.associated_bs = None
#             ue.sinr = -np.inf
#             ue.ewma_dr = 0.0
#         return self._get_obs(), {}    

#     # Add these to your NetworkEnvironment class
#     def calculate_jains_fairness(self):
#         throughputs = [ue.throughput for ue in self.ues]
#         return (sum(throughputs) ** 2) / (len(throughputs) * sum(t**2 for t in throughputs))

#     @property
#     def throughput(self):
#         return torch.log2(1 + 10**(self.sinr/10)).item()
        
#     def calculate_reward(self):
#         """Calculate reward with tensor-safe operations"""
#         # Convert SINR values to tensor first
#         sinr_tensor = torch.tensor([ue.sinr for ue in self.ues], dtype=torch.float32)
        
#         throughput = torch.sum(torch.log2(1 + 10**(sinr_tensor/10)))
#         loads = torch.tensor([bs.load for bs in self.base_stations])
        
#         jain = (loads.sum()**2) / (self.num_bs * (loads**2).sum() + 1e-6)
#         overload_penalty = torch.sum(torch.relu(loads - 1.0))
        
#         return throughput + 2.0 * jain - 0.5 * overload_penalty
    
#     def _admit_nearest_ue(self, bs):
#         """Find and associate nearest unassociated UE to BS"""
#         # Get all unassociated UEs
#         unassociated_ues = [ue for ue in self.ues if ue.associated_bs is None]
        
#         if not unassociated_ues:
#             return
        
#         # Calculate distances using vector math
#         bs_pos = bs.position.numpy()
#         ue_positions = np.array([ue.position.numpy() for ue in unassociated_ues])
        
#         distances = np.linalg.norm(ue_positions - bs_pos, axis=1)
#         nearest_idx = np.argmin(distances)
#         nearest_ue = unassociated_ues[nearest_idx]
        
#         # Associate UE
#         nearest_ue.associated_bs = bs.id
#         bs.allocated_resources[nearest_ue.id] = nearest_ue.demand

#     def _offload_most_demanding_ue(self, bs):
#         """Remove UE with highest demand from BS"""
#         if not bs.allocated_resources:
#             return
        
#         # Find UE with max demand
#         ue_id = max(bs.allocated_resources, key=lambda k: bs.allocated_resources[k])
#         ue = self.ues[ue_id]
        
#         # Disassociate
#         del bs.allocated_resources[ue_id]
#         ue.associated_bs = None   
        
#     # def step(self, actions: Dict[str, int]):
#     #     # 1. Process UE actions (BS selections)
#     #     for agent_id, bs_choice in actions.items():
#     #         ue_id = int(agent_id.split("_")[1])
#     #         ue = self.ues[ue_id]
#     #         target_bs = self.base_stations[bs_choice]
            
#     #         # Disassociate from old BS (if any)
#     #         if ue.associated_bs is not None:
#     #             old_bs = self.base_stations[ue.associated_bs]
#     #             if ue.id in old_bs.allocated_resources:
#     #                 del old_bs.allocated_resources[ue.id]
            
#     #         # Associate with new BS if capacity allows
#     #         if target_bs.load + ue.demand <= target_bs.capacity:
#     #             target_bs.allocated_resources[ue.id] = ue.demand
#     #             ue.associated_bs = bs_choice
        
#     #     # 2. Update BS loads and UE SINRs
#     #     for bs in self.base_stations:
#     #         bs.load = sum(bs.allocated_resources.values())
        
#     #     # 3. Calculate rewards (individual + global)
#     #     rewards = {}
#     #     # global_fairness = self.calculate_jains_fairness()  # From earlier code
#     #     for ue in self.ues:
#     #         if ue.associated_bs is None:
#     #             reward = -1.0  # Penalize unconnected UEs
#     #         else:
#     #             bs = self.base_stations[ue.associated_bs]
#     #             throughput = np.log2(1 + 10**(ue.sinr/10))
#     #             # reward = throughput + 0.5 * global_fairness  # Hybrid reward
#     #             reward = self.calculate_individual_reward()
            
#     #         rewards[f"ue_{ue.id}"] = reward
        
#     #     # 4. Check termination
#     #     self.current_step += 1
#     #     done = {"__all__": self.current_step >= self.episode_length}
        
#     #     return self._get_obs(), rewards, done, {}
    
#     def _calculate_sinrs(self, ue):
#         sinrs = []
#         for bs in self.base_stations:
#             signal = bs.received_power_mW(ue.position)
#             # use the method you wrote to compute interference at the BS—
#             # here we just compute interference *at the UE*, so we inline it
#             interf = sum(
#                 other.received_power_mW(ue.position)
#                 for other in self.base_stations
#                 if other is not bs
#             )
#             noise = bs.noise_mW()
#             sinrs.append(10 * np.log10(signal/(interf+noise+1e-12)))
#         return np.clip(np.array(sinrs), -30,60)

    
#     def _update_sinrs(self):
#         for ue in self.ues:
#             if ue.associated_bs is not None:
#                 sinrs = self._calculate_sinrs(ue)
#                 ue.sinr = sinrs[ue.associated_bs]
#             else:
#                 ue.sinr = -np.inf
    
    
#     def step(self, actions: Dict[str, int]):
#         # Add timing for performance analysis
#         print(f"Step called with {len(actions)} actions")
#         try:
#             start_time = time.time()

#             # 1) Clear old allocations & reset loads
#             for bs in self.base_stations:
#                 bs.allocated_resources.clear()
#                 bs.calculate_load()

#             # 2) Process UE associations (sorted by demand for packing)
#             ue_actions = sorted(
#                 ((int(aid.split("_")[1]), bs_choice) for aid, bs_choice in actions.items()),
#                 key=lambda x: self.ues[x[0]].demand
#             )
#             bs_allocations = {bs.id: 0 for bs in self.base_stations}
#             bs_capacity_used = {bs.id: 0.0 for bs in self.base_stations}
#             connected_count = 0
            
#             # assume ue_actions is a list of (ue_id, _) sorted by demand
#             for ue_id, _ in ue_actions:
#                 ue = self.ues[ue_id]

#                 # 1) primary choice by PF+load score
#                 primary = self.pick_best_bs(ue)

#                 # 2) build ordered list: primary first, then the rest
#                 candidates = [primary] + [i for i in range(self.num_bs) if i != primary]

#                 # 3) try each candidate until one fits
#                 admitted = False
#                 for bs_idx in candidates:
#                     bs = self.base_stations[bs_idx]
#                     dr = bs.data_rate_shared(ue)               # how much rate UE would use
#                     if bs.load + dr <= bs.capacity:            # capacity check
#                         bs.allocated_resources[ue.id] = dr
#                         bs.calculate_load()                    # update bs.load
#                         ue.associated_bs = bs_idx
#                         admitted = True
#                         break

#                 # 4) if none could admit, leave unassociated
#                 if not admitted:
#                     ue.associated_bs = None

            
#             # 3) Update SINR & EWMA
#             for ue in self.ues:
#                 # full multi-cell SINR vector
#                 sinr_vec = self._calculate_sinrs(ue)
#                 if ue.associated_bs is not None:
#                     ue.sinr = sinr_vec[ue.associated_bs]
#                     # update EWMA with allocated rate
#                     self.base_stations[ue.associated_bs].calculate_load()
#                     measured = self.base_stations[ue.associated_bs].allocated_resources.get(ue.id, 0.0)
#                     ue.update_ewma(measured)
#                 else:
#                     ue.sinr = -np.inf
            
#             # Calculate rewards efficiently
#             rewards = {}
#             total_reward = 0
#             for bs in self.base_stations:
#                 print(f"BS {bs.id}: load={bs.load}, capacity={bs.capacity}, ratio={bs.load/bs.capacity if bs.capacity>0 else 'inf'}")
#             # 4) Compute rewards and aggregate metrics
#             rewards = {}
#             total_reward = 0.0
#             for ue in self.ues:
#                 key = f"ue_{ue.id}"
#                 r = self.calculate_individual_reward(key)
#                 rewards[key] = r
#                 total_reward += r                 
                
#                 # if ue_id % 10 == 0:  # Only print every 10th UE to avoid excessive output
#                 #         print(f"UE {ue_id}: connected={ue.associated_bs is not None}, "
#                 #             f"sinr={ue.sinr:.2f}, throughput={throughput:.2f}, "
#                 #             f"load_factor={load_factor:.2f}, reward={rewards[f'ue_{ue.id}']:.4f}")
                        
#             print(f"Connected Users : {connected_count} Users")
#             # Log performance metrics at regular intervals
#             step_time = time.time() - start_time
#             if self.current_step % 10 == 0 or connected_count < self.num_ue:
#                 print(f"Step {self.current_step} | Time: {step_time:.3f}s | Connected: {connected_count}/{self.num_ue} UEs")
                
#                 # Calculate load balancing metrics
#                 loads = [bs.load for bs in self.base_stations]
#                 capacities = [bs.capacity for bs in self.base_stations]
#                 utilizations = [load/cap if cap > 0 else 0 for load, cap in zip(loads, capacities)]
                
#                 # Jain's fairness index for load distribution
#                 if sum(utilizations) > 0:
#                     squared_sum = sum(utilizations)**2
#                     sum_squared = sum(u**2 for u in utilizations) * len(utilizations)
#                     jains_index = squared_sum / sum_squared if sum_squared > 0 else 0
#                     print(f"Load Balancing Fairness: {jains_index:.4f}")
#                     # jain = self.calculate_jains_fairness()
#                     # print(f"Load balancing Jain's index: {jain:.4f}")
                
#                 # Log detailed BS allocations if not all UEs connected
#                 if connected_count < self.num_ue:
#                     print(f"BS allocations: {bs_allocations}")
#                     print(f"BS capacity %: {[bs_capacity_used[bs.id]/bs.capacity if bs.capacity > 0 else 0 for bs in self.base_stations]}")
            
            
            
#             # Track current solution for visualization
#             current_solution = []
#             for ue in self.ues:
#                 if ue.associated_bs is not None:
#                     current_solution.append(ue.associated_bs)
#                 else:
#                     # Use a default value for unconnected UEs
#                     current_solution.append(0)  # Or None if your visualization can handle it
#             print(f"Current solution at : {current_solution}")
#             sinr_list = [ue.sinr if ue.associated_bs is not None else -np.inf
#              for ue in self.ues]
            
#             # KPI logging
#             if self.log_kpis and self.kpi_logger:
#                 metrics = {
#                     "connected_ratio": connected_count/self.num_ue,
#                     "step_time": step_time,
#                     "episode_reward_mean": total_reward/self.num_ue,
#                     "fairness_index": jain,
#                     "throughput": sum(np.log2(1+10**(ue.sinr/10)) for ue in self.ues if ue.associated_bs is not None),
#                     "solution":       current_solution,
#                     "sinr_list":      sinr_list,
#                 }
#                 self.kpi_logger.log_metrics(
#                     phase="environment", algorithm="hybrid_marl",
#                     metrics=metrics, episode=self.current_step
#                 )
                
#             # Split termination into terminated and truncated (for Gymnasium compatibility)
#             terminated = {"__all__": False}  # Episode is not terminated due to failure condition
#             truncated = {"__all__": self.current_step >= self.episode_length}  # Episode length limit reached    
#             # Common info for all agents
#             common_info = {
#                 "connected_ratio": connected_count / self.num_ue,
#                 "step_time": step_time,
#                 "current solution":current_solution,
#                 "avg_reward": total_reward / self.num_ue if self.num_ue > 0 else 0
#             }
            
#             # Create info dict with one entry per agent, plus common info
#             info = {
#                 f"ue_{ue.id}": {
#                     "connected": ue.associated_bs is not None,
#                     "sinr": float(ue.sinr)
#                 } for ue in self.ues  # Add minimal agent-specific info
#             }
#             info["__common__"] = common_info  # Add common info under special key
#             # Save as last info
#             self.last_info = info
#             # Increment step counter
#             self.current_step += 1
#             return self._get_obs(), rewards, terminated, truncated, info
        
#         except Exception as e:
#             print(f"ERROR in step: {e}")
#             import traceback
#             print(traceback.format_exc())
#             # Return a safe default response
#             return self._get_obs(), {f"ue_{ue.id}": 0.0 for ue in self.ues}, {"__all__": False}, {"__all__": True}, {"__common__": {"error": str(e)}}
        
#     # Add this to your NetworkEnvironment class
#     def get_last_info(self):
#         """Return the last info dict from a step"""
#         if hasattr(self, 'last_info'):
#             print("Getting lastest info....")
#             return self.last_info
#         return None
#     # def _get_obs(self):
#     #     # Precompute BS loads and positions once
#     #     bs_loads = np.array([bs.load for bs in self.base_stations], dtype=np.float32)
#     #     bs_positions = np.array([bs.position for bs in self.base_stations], dtype=np.float32)
                
#     #     obs = {}
#     #     for ue in self.ues:
#     #         # Calculate SINR to all BSs (vectorized)
#     #         distances = np.linalg.norm(bs_positions - ue.position, axis=1)
#     #         sinrs = 30.0 - 20 * np.log10(distances + 1e-6)  # Simplified model
            
#     #         # Store SINR for the associated BS (if any)
#     #         if ue.associated_bs is not None:
#     #             ue.sinr = sinrs[ue.associated_bs]  # Add this line
#     #         else:
#     #             ue.sinr = -np.inf  # No connection
            
#     #         # Normalize features
#     #         normalized_sinrs = sinrs / 40.0  # Assume max SINR ~40 dB
#     #         normalized_loads = bs_loads / 1000.0  # Assuming max load=1000
                        
#     #         obs[f"ue_{ue.id}"] = np.concatenate([
#     #             normalized_sinrs,
#     #             normalized_loads,
#     #             [ue.demand / 200.0]  # Max demand=200
#     #         ], dtype=np.float32)
            
            
#     #     return obs
    
#     def _get_obs(self):        
#         # Precompute BS loads, capacities, positions
#         bs_loads = np.array([bs.load for bs in self.base_stations], dtype=np.float32)
#         bs_caps  = np.array([bs.capacity for bs in self.base_stations], dtype=np.float32)
#         bs_positions = np.stack([bs.position for bs in self.base_stations])

#         # Normalize loads by capacity
#         normalized_loads = bs_loads / (bs_caps + 1e-9)
#         # Utilization = clipped load/capacity
#         bs_utils = np.clip(bs_loads / (bs_caps + 1e-9), 0.0, 1.0).astype(np.float32)

#         obs = {}
#         for ue in self.ues:
#             # Compute full SINR vector (interference + noise)
#             sinr_vec = self._calculate_sinrs(ue)
#             # Record UE.sinr
#             ue.sinr = sinr_vec[ue.associated_bs] if ue.associated_bs is not None else -np.inf

#             # Normalize SINRs (assume max ~40 dB)
#             normalized_sinrs = sinr_vec / 40.0

#             # Normalize own demand
#             norm_demand = np.array([ue.demand / 200.0], dtype=np.float32)

#             # Concatenate into one feature vector
#             obs[f"ue_{ue.id}"] = np.concatenate([
#                 normalized_sinrs.astype(np.float32),
#                 normalized_loads,
#                 bs_utils,
#                 norm_demand
#             ], axis=0)

#         return obs


#     def _calculate_local_interference(self, bs):
#         """Calculate interference from neighboring BSs"""
#         neighbor_dist = 30  # Consider BSs within 30 units
#         interference = 0.0
        
#         for other_bs in self.base_stations:
#             if other_bs.id == bs.id:
#                 continue
                
#             distance = np.linalg.norm(bs.position - other_bs.position)
#             if distance < neighbor_dist:
#                 interference += other_bs.transmit_power / (distance ** 2 + 1e-6)
        
#         return np.log1p(interference)  # Compress scale
        

#     def _get_normalized_ue_positions(self, bs):
#         """Normalize positions to [-1,1] range"""
#         positions = []
#         for ue in self.ues:
#             rel_pos = (ue.position - bs.position).numpy()
#             # Normalize based on environment bounds (0-100 in your case)
#             norm_pos = rel_pos / 50.0 - 1.0  # Scales to [-1,1]
#             positions.append(norm_pos.astype(np.float32))  # ✅ Cast
#         return np.array(positions).flatten()
    
#     # In NetworkEnvironment.get_current_state() for visualization
#     def get_current_state(self):
#         return {
#             "base_stations": [
#                 {
#                     "id": bs.id,
#                     "position": bs.position.tolist(),  # bs.position.numpy(),
#                     "load": bs.load,
#                     "capacity": bs.capacity
#                 } for bs in self.base_stations
#             ],
#             "users": [
#                 {
#                     "id": ue.id,
#                     "position": ue.position.tolist(),  # ue.position.numpy(),
#                     "associated_bs": ue.associated_bs,
#                     "sinr": ue.sinr.item()  # Convert tensor to float
#                 } for ue in self.ues
#             ],
#             "associations": self.associations.copy(),
#             "version": self.version,
#             # "metaheuristic_agents": self.current_metaheuristic_agents  # Set by optimizer
#         }
        
#     def get_state_snapshot(self) -> dict:
#         return {
#             "users": [{
#                 "id": ue.id,
#                 "position": np.array(ue.position, copy=True),
#                 "velocity": np.array(ue.velocity, copy=True),
#                 "demand": ue.demand,
#                 "associated_bs": ue.associated_bs,
#                 "sinr": float(ue.sinr)
#             } for ue in self.ues],
#             "base_stations": [{
#                 "id": bs.id,
#                 "allocated_resources": bs.allocated_resources.copy(),
#                 "load": bs.load,
#                 "capacity": bs.capacity
#             } for bs in self.base_stations],
#             "current_step": self.current_step
#         }

#     def set_state_snapshot(self, state: dict):
#         for ue_state in state["users"]:
#             ue = next(u for u in self.ues if u.id == ue_state["id"])
#             ue.position = np.array(ue_state["position"], copy=True)
#             ue.velocity = np.array(ue_state["velocity"], copy=True)
#             ue.demand = ue_state["demand"]
#             ue.associated_bs = ue_state["associated_bs"]
#             ue.sinr = float(ue_state["sinr"])
#         for bs_state in state["base_stations"]:
#             bs = next(b for b in self.base_stations if b.id == bs_state["id"])
#             bs.allocated_resources = bs_state["allocated_resources"].copy()
#             bs.load = bs_state["load"]
#             bs.capacity = bs_state["capacity"]
#         self.current_step = state["current_step"]

    
#     def _update_system_metrics(self):
#         # 1) Recompute loads from PF‐shared allocations
#         for bs in self.base_stations:
#             bs.calculate_load()

#         # 2) Recompute each UE’s SINR using the multi‐cell helper
#         for ue in self.ues:
#             if ue.associated_bs is not None:
#                 sinr_vec = self._calculate_sinrs(ue)     # NumPy vector of SINRs
#                 ue.sinr = sinr_vec[ue.associated_bs]
#             else:
#                 ue.sinr = -np.inf

    
#     def apply_solution(self, solution):
#         """Apply a solution (numpy array or dict) to the environment"""
#         # 1) Unwrap array into {bs_id:[ue_ids]} if needed
#         if isinstance(solution, np.ndarray):
#             sol_dict = {bs.id: [] for bs in self.base_stations}
#             for ue_idx, bs_id in enumerate(solution.astype(int)):
#                 sol_dict[int(bs_id)].append(ue_idx)
#             solution = sol_dict

#         # Validate BS IDs exist
#         valid_bs_ids = {int(bs.id) for bs in self.base_stations}
#         for bs_id in solution.keys():
#             try:
#                 bs_id_int = int(bs_id)
#                 if bs_id_int not in valid_bs_ids:
#                     raise ValueError(f"Invalid BS ID {bs_id} in solution")
#             except ValueError:
#                 raise ValueError(f"BS ID {bs_id} is not an integer")
            
#         # Optional: Validate that each user index is within range.
#         num_ues = len(self.ues)
#         for bs_id, ue_ids in solution.items():
#             for ue_id in ue_ids:
#                 if ue_id < 0 or ue_id >= num_ues:
#                     raise IndexError(f"UE index {ue_id} for BS ID {bs_id} is out of valid range (0 to {num_ues-1}).")
        
#         # Clear existing associations
#         for bs in self.base_stations:
#             # bs.allocated_resources = {}
#             bs.allocated_resources.clear()
#             bs.calculate_load()
#             self.associations[bs.id] = []

#         ## 4) Assign PF‐shared rates, not raw demand
#         for bs_id, ue_list in solution.items():
#             bs = next(b for b in self.base_stations if b.id == int(bs_id))
#             for ue_id in ue_list:
#                 ue = self.ues[ue_id]
#                 dr = bs.data_rate_shared(ue)               # PF‐shared rate (NumPy)
#                 bs.allocated_resources[ue.id] = dr
#             bs.calculate_load()

#         # 5) Refresh SINR metrics
#         self._update_system_metrics()
    
    
    
    
#     def evaluate_detailed_solution(self, solution, alpha=0.1, beta=0.1):
#         """
#         Apply a candidate user‑association solution, compute detailed performance metrics, then restore state.
#         """
#         # 1) Snapshot current state
#         original = self.get_state_snapshot()

#         # 2) Apply the proposed associations
#         self.apply_solution(solution)
#         connected_count = sum(1 for ue in self.ues if ue.associated_bs is not None)
#         print(f"Connected UEs = {connected_count} / {self.num_ue}")
        
#         # 3) Ensure loads and SINRs are up-to-date
#         for bs in self.base_stations:
#             bs.calculate_load()
#         for ue in self.ues:
#             ue.sinr = self._calculate_sinrs(ue)[ue.associated_bs] if ue.associated_bs is not None else -np.inf

#         # 4) Throughputs per UE (linear units)
#         throughputs = np.array([
#             np.log2(1 + 10**(ue.sinr / 10)) if ue.associated_bs is not None else 0.0
#             for ue in self.ues
#         ], dtype=np.float32)

#         # 5) Compute metrics
#         fitness = self.calculate_reward()  # global environment reward
#         avg_sinr = np.mean([ue.sinr for ue in self.ues])
#         avg_throughput = throughputs.mean()
#         fairness = (throughputs.sum()**2) / (len(throughputs) * np.sum(throughputs**2) + 1e-9)
#         load_var = np.var([bs.load for bs in self.base_stations])
#         bs_loads = [bs.load for bs in self.base_stations]

#         # 6) Restore original state
#         self.set_state_snapshot(original)

#         # 7) Return detailed report
#         return {
#             "fitness": float(fitness),
#             "average_sinr": float(avg_sinr),
#             "average_throughput": float(avg_throughput),
#             "fairness": float(fairness),
#             "load_variance": float(load_var),
#             "bs_loads": bs_loads
#         }


        
# # env = NetworkEnvironment({"num_ue": 3, "num_bs": 2})
# # obs, _ = env.reset()
# # print(obs["ue_0"].shape)  # Should be (2*2 + 1)=5

# # actions = {"ue_0": 1, "ue_1": 0, "ue_2": 1}  # Each UE selects a BS index
# # next_obs, rewards, dones, _ = env.step(actions)
# # print(next_obs, rewards, dones, _ )