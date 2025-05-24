        
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
import math
import numpy as np

class PolicyMappingManager:
    def __init__(self, bs_positions: np.ndarray, initial_ue_positions: Dict[str, np.ndarray]):
        """
        Initialize policy mapping manager
        
        Args:
            bs_positions: Array of shape (num_bs, 2) with BS positions
            initial_ue_positions: Dict mapping agent_id to initial position
        """
        self.bs_positions = bs_positions
        self.ue_positions = initial_ue_positions.copy()
        
    def update_ue_positions(self, new_positions: Dict[str, np.ndarray]):
        """Update UE positions when they move"""
        self.ue_positions.update(new_positions)
        
    def get_closest_bs(self, agent_id: str) -> int:
        """
        Find closest BS to a UE based on current tracked positions
        
        Args:
            agent_id: UE identifier (e.g., "ue0")
            
        Returns:
            BS index (0 = macro, 1-3 = small cells)
        """
        # Wherever you're setting up ue_positions
        # Extract the numeric part from "ue_0" -> 0
               
            
        if agent_id not in self.ue_positions:
            print(f"Warning: {agent_id} position not found, defaulting to macro")
            return 0
            
        ue_pos = self.ue_positions[agent_id]
        
        # Calculate distances to all BSs
        distances = [
            np.linalg.norm(ue_pos - bs_pos) 
            for bs_pos in self.bs_positions
        ]
        
        return int(np.argmin(distances))
    
    def get_policy_distribution(self) -> Dict[str, int]:
        """Get count of UEs assigned to each policy"""
        distribution = {"bs_0_policy": 0, "bs_1_policy": 0, "bs_2_policy": 0, "bs_3_policy": 0}
        
        for agent_id in self.ue_positions.keys():
            closest_bs = self.get_closest_bs(agent_id)
            policy_name = f"bs_{closest_bs}_policy"
            distribution[policy_name] += 1
            
        return distribution
    
    def log_policy_assignments(self):
        """Log current policy assignments for debugging"""
        assignments = {}
        for agent_id in self.ue_positions.keys():
            closest_bs = self.get_closest_bs(agent_id)
            assignments[agent_id] = f"bs_{closest_bs}_policy"
        
        distribution = self.get_policy_distribution()
        # print(f"Policy distribution: {distribution}")
        return assignments
class UE:
    def __init__(self, id, position, demand,
                v_min=0.5, v_max=1.5,
                pause_min=1.0, pause_max=5.0,rx_gain_dbi=0.0,
                ewma_alpha=0.7):
        self.id          = int(id)
        self.position    = np.array(position, dtype=np.float32)
        self.demand      = float(demand)       # Mbps
        # MRWP fields:
        self.v_min       = v_min               # min speed (m/s)
        self.v_max       = v_max               # max speed (m/s)
        self.pause_min   = pause_min           # min pause (s)
        self.pause_max   = pause_max           # max pause (s)
        self.waypoint    = self._draw_waypoint()
        self.speed       = np.random.uniform(v_min, v_max)
        self.pause_time  = 0.0                 # start “moving”
        self.seed = 42
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # Scheduling fields:
        
    
        self.associated_bs = None
        self.sinr          = -np.inf
        self.ewma_dr       = 1e6
        self.ewma_alpha    = ewma_alpha
        self.rx_gain = float(rx_gain_dbi)  # dBi
    def _draw_waypoint(self):
        # uniformly anywhere in the 100×100 area
        return np.random.uniform(0, 100, size=2).astype(np.float32)

    def update_position(self, delta_time=0.08):
        """MRWP: if paused, count down; else move toward waypoint."""
        if self.pause_time > 0:
            # still in pause
            self.pause_time -= delta_time
            return

        # vector and distance to waypoint
        direction = self.waypoint - self.position
        dist      = np.linalg.norm(direction)
        # how far we’d travel this step
        travel    = self.speed * delta_time

        if travel >= dist:
            # reached (or overshot) the waypoint
            self.position   = self.waypoint.copy()
            # draw a new pause interval
            self.pause_time = np.random.uniform(self.pause_min, self.pause_max)
            # pick next waypoint & speed
            self.waypoint   = self._draw_waypoint()
            self.speed      = np.random.uniform(self.v_min, self.v_max)
        else:
            # move fractionally toward the waypoint
            self.position += (direction / dist) * travel

    def update_ewma(self, measured_dr):
        self.ewma_dr = self.ewma_alpha * self.ewma_dr \
                     + (1 - self.ewma_alpha) * measured_dr
    
    # def update_ewma(self, allocated_rb):
    #     # EWMA of RBs per TTI
    #     self.ewma_rb = self.ewma_alpha * self.ewma_rb + (1 - self.ewma_alpha) * allocated_rb
                     
    def get_required_rbs(self, bs):
        """Calculate how many RBs this UE needs from this BS"""
        sinr = self._calculate_sinrs(bs, 0)  # Using RB 0 as reference
        spectral_efficiency = np.log2(1 + sinr)  # bits/s/Hz
        rb_capacity = bs.rb_bandwidth * 1e6 * spectral_efficiency  # bits/s per RB
        
        # RBs needed to satisfy demand (bps)
        return int(np.ceil((self.demand * 1e6) / rb_capacity))
                    
class BaseStation:
    def __init__(self, id, position, frequency, bandwidth, height=50.0, reuse_color=None,tx_power_dbm=30.0,tx_gain_dbi=8.0,
                subcarrier_spacing=60e3, bf_gain_dbi=20.0, path_loss_n=2.0,path_loss_sigma=0.0,cre_bias=0.0):

        self.id = int(id)
        self.position = np.array(position, dtype=np.float32)
        self.frequency = float(frequency)    # Hz
        self.bandwidth = float(bandwidth)    # Hz
        self.height = float(height)          # m
        self.tx_power = float(tx_power_dbm)  # dBm
        self.tx_gain       = float(tx_gain_dbi)  # dBi
        self.bf_gain      = bf_gain_dbi   # beamforming gain
        self.allocated_resources = {}        # {ue_id: allocated_rate}
        self.load = 0.0                      # sum of allocated rates
        self.capacity = 0 # self.bandwidth * 0.8 # e.g. 80% of bandwidth in Mbps
        self.reuse_color   = reuse_color         # e.g. "A", "B", or "C"
        self.path_loss_n     = float(path_loss_n)
        self.path_loss_sigma = float(path_loss_sigma)
        self.cre_bias        = float(cre_bias)
        # Resource block parameters
        self.rb_bandwidth = 12*subcarrier_spacing # Hz (typical 5G/6G subcarrier spacing × 12)
        self.num_rbs = int(self.bandwidth / self.rb_bandwidth)  # Number of available RBs
        # print(f"BS with reuse Colour: {reuse_color},RB Bandwidth: {self.rb_bandwidth}, No. Num_rbs:{self.num_rbs}")
        self.seed = 42
        if self.seed is not None:
            np.random.seed(self.seed)
        
        self.rb_allocation = {}  # Dictionary mapping UE IDs to allocated RBs
        self.rb_sinr = np.zeros(self.num_rbs)  # SINR per RB (for interference modeling)
        
    def calculate_load(self):
        # total number of RBs in use:
        used_rbs = sum(len(rbs) for rbs in self.rb_allocation.values())
        # if you want absolute count:
        self.load = used_rbs

    # def path_loss(self, distance, ue_height=1.5):
    #     # Okumura-Hata suburban model
    #     f = self.frequency
    #     hb = self.height
    #     hu = ue_height
    #     ch = 0.8 + (1.1 * np.log10(f) - 0.7) * hu - 1.56 * np.log10(f)
    #     const1 = 69.55 + 26.16 * np.log10(f) - 13.82 * np.log10(hb) - ch
    #     const2 = 44.9 - 6.55 * np.log10(hb)
    #     return const1 + const2 * np.log10(distance + 1e-9)
    
    def path_loss(self, distance, d0=1.0):
        """
        Close-In Reference Distance Path-Loss Model.
        distance: link distance in meters
        d0: reference distance (m), typically 1.0
        n: path-loss exponent (environment-specific)
        sigma: shadow-fading std. dev. (dB)
        """
        c = 3e8  # speed of light (m/s)

        # 1) Free‐space loss at d0 (usually 1 m):
        #    FSPL(d0) = 20 log10(4π d0 f / c)
        
        fspl_d0 = 20*np.log10(4*np.pi*d0*self.frequency/c)
        # 2) Distance‐dependent term
        
        pl_mean = fspl_d0 + 10*self.path_loss_n*np.log10(distance/d0)
        # 3) Add shadow fading if desired
        
        shadow   = np.random.randn()*self.path_loss_sigma
        return pl_mean + shadow
    

    def noise_mW(self):
        # Thermal noise: -174 dBm/Hz + 10*log10(BW_Hz)
        noise_dbm = -174 + 10 * np.log10(self.bandwidth )
        return 10 ** (noise_dbm / 10)
    
    
    def _calculate_local_interference(self, neighbor_dist=None):
        # Set default neighbor distance by tier
        if neighbor_dist is None:
            neighbor_dist = 1000.0 if self.reuse_color=="Macro" else 100.0

        interference = 0.0
        for other in self.base_stations:
            if other.id == self.id:
                continue

            d = np.linalg.norm(self.position - other.position)
            if d > neighbor_dist:
                continue

            # other.received_power_mW already includes path-loss
            interference += other.received_power_mW(self.position)

        return interference


    
    # def received_power_mW(self, ue_pos, ue_rx_gain=None):# , ue_bf_gain_dbi=0.0
    #     d = np.linalg.norm(self.position - ue_pos)
    #     # The code is calculating the path loss using a function `path_loss` with the distance `d` as
    #     # an input parameter and storing the result in the variable `L`.
    #     L  = self.path_loss(d)
    #     G_tx = self.tx_gain + self.bf_gain      # total transmit gain
    #     G_rx = ue_rx_gain if ue_rx_gain is not None else 0.0 # (ue_rx_gain or 0.0) + ue_bf_gain_dbi  # UE may also beam-form
    #     p_rx_dbm = self.tx_power + G_tx + G_rx - L
    #     return 10**(p_rx_dbm/10)
    def received_power_mW(self, ue_position, ue_rx_gain=None):
        """Calculate received power with atmospheric/environmental losses"""
        # 1. Basic free space path loss
        distance = np.linalg.norm(ue_position - self.position)
        lambda_ = 3e8 / (self.frequency * 1e6)  # Wavelength in meters
        
        # Free space path loss (dB)
        fspl = 20 * np.log10(distance) + 20 * np.log10(self.frequency) + 20 * np.log10(4 * np.pi / 3e8)
        
        # 2. Atmospheric attenuation components (ITU-R models)
        def rain_attenuation(freq_ghz, rain_rate, distance_km):
            """ITU-R P.838-8 specific attenuation due to rain"""
            k = 0.0335 * (freq_ghz**0.9)  # Coefficients for 10-100 GHz
            alpha = 1 - 0.02 * (freq_ghz - 10)
            return k * (rain_rate**alpha) * distance_km

        def gaseous_attenuation(freq_ghz, temp=15, humidity=60):
            """ITU-R P.676-13 oxygen/water vapor absorption"""
            # Simplified model for 1-60 GHz
            if freq_ghz < 15:
                return 0.05 * freq_ghz  # dB/km
            elif 15 <= freq_ghz < 60:
                return 0.1 * (freq_ghz - 14) + 0.75  # dB/km
            else:
                return 3.0  # dB/km (approximate for 60+ GHz)

        # 3. Environmental parameters (customize these)
        rain_rate = 10  # mm/h - moderate rain
        humidity = 60    # %
        temperature = 25  # °C
        
        # Calculate additional losses
        distance_km = distance / 1000
        freq_ghz = self.frequency / 1e3
        
        L_rain = rain_attenuation(freq_ghz, rain_rate, distance_km)
        L_gas = gaseous_attenuation(freq_ghz, temperature, humidity) * distance_km
        L_cloud = 0.1 * distance_km  # Simple cloud attenuation
        
        # 4. Antenna and system losses
        L_polarization = 1  # dB - polarization mismatch
        L_pointing = 0.5    # dB - beam misalignment
        
        # 5. Total additional losses
        total_add_loss = L_rain + L_gas + L_cloud + L_polarization + L_pointing
        
        # 6. Final received power calculation
        Prx_dBm = (self.tx_power
                + self.tx_gain 
                - fspl 
                - total_add_loss)
        # Macro-specific fixed attenuation (IN dB DOMAIN)
        if self.reuse_color == "Macro":
            fixed_macro_loss = 50  # dB
            Prx_dBm -= fixed_macro_loss  # Proper dB subtraction
        else:
            fixed_macro_loss = 5  # dB
            Prx_dBm -= fixed_macro_loss  # Proper dB subtraction
            
        return 10 ** (Prx_dBm / 10)  # Convert dBm to mW
    
    def snr_linear(self, ue, cross_tier=True, max_sinr_db=40.0):
        """
        Compute the linear SINR for this BS → ue link, including
        Tx/Rx gains, co-channel (or cross-tier) interference, and thermal noise.
        
        cross_tier: if True, include interference from all BSs; 
                    if False, only from same reuse_color.
        max_sinr_db: cap SINR at this dB value (e.g. 50 dB).
        """

        # 1) desired signal power (mW)
        signal_mW = self.received_power_mW(ue.position, ue.rx_gain)

        # 2) build interference sum
        interference_mW = 0.0
        for other in self.base_stations:
            if other.id == self.id:
                continue

            if not cross_tier:
                # only co-channel
                if other.reuse_color != self.reuse_color:
                    continue

            # include this BS’s contribution
            interference_mW += other.received_power_mW(ue.position, ue.rx_gain)

        # 3) noise power (mW)
        noise_mW = self.noise_mW()

        # 4) raw linear SINR
        lin_snr = signal_mW / (interference_mW + noise_mW + 1e-12)

        # 5) apply cap
        snr_cap_linear = 10 ** (max_sinr_db / 10)
        return min(lin_snr, snr_cap_linear)

    
    def allocate_prbs(self):
        """
        Improved version of the PRB allocation method that ensures fairer distribution
        and resolves bugs in the original implementation.
        """
        # Handle empty case
        if not self.ues:
            self.rb_allocation = {}
            self.allocated_resources = {}
            self.calculate_load()
            return
            
        # Determine UE IDs properly
        if isinstance(self.ues, dict):
            ue_ids = list(self.ues.keys())
        else:
            # self.ues is a list of UE objects
            ue_ids = [ue.id for ue in self.ues]

        self.rb_allocation = {ue_id: [] for ue_id in ue_ids}
        delivered_bits = {ue_id: 0.0 for ue_id in ue_ids}

        # Fast path for single UE case
        if len(self.ues) == 1:
            ue_id = next(iter(self.ues))
            ue = self.ues[ue_id]
            max_bits = ue.demand * 1e6  # demand in bits/sec

            # Tentatively give it all PRBs
            prb_list = list(range(self.num_rbs))
            sinr_array = self.snr_per_rb(ue)
            rate_array = self.rb_bandwidth * np.log2(1 + sinr_array)
            total_bits = rate_array.sum()

            # Cap at demand
            allocated_bits = min(total_bits, max_bits)
            # Figure out how many PRBs to use to stay under demand
            if allocated_bits < total_bits:
                # allocate highest‐rate PRBs until sum(rate) ≥ max_bits
                order = np.argsort(-rate_array)
                cum = 0.0
                prb_list = []
                for prb in order:
                    cum += rate_array[prb]
                    prb_list.append(prb)
                    if cum >= max_bits:
                        break
            # Store the allocation
            self.rb_allocation[ue_id] = prb_list
            delivered_bits[ue_id] = min(max_bits, sum(rate_array[prb] for prb in prb_list))
            # skip the rest of the algorithm
            self.allocated_resources = {ue_id: delivered_bits[ue_id]}
            self.calculate_load()
            return
        
        # --------------------------------------------------
        # OPTIMIZATION 1: Efficient Precomputation with Caching
        # --------------------------------------------------
        
        # Precompute SINR arrays, rates, and best PRB ordering for each UE
        sinr_map = {}               # Store raw SINR values
        sinr_db_map = {}            # Store SINR in dB for clustering
        rate_map = {}               # Store achievable rates per PRB
        sorted_prb_indices = {}     # Store pre-sorted PRB indices (best first)
        avg_sinr_db = {}            # Store average SINR in dB per UE
        
        for ue_id, ue in self.ues.items():
            # Get SINR for all PRBs at once (single calculation)
            sinr_array = self.snr_per_rb(ue)
            sinr_map[ue_id] = sinr_array
            
            # Calculate SINR in dB for clustering
            sinr_db_array = 10 * np.log10(sinr_array + 1e-10)  # Avoid log of zero
            sinr_db_map[ue_id] = sinr_db_array
            
            # Precalculate rates for all PRBs
            rate_map[ue_id] = self.rb_bandwidth * np.log2(1 + sinr_array)
            
            # Calculate average SINR for clustering
            avg_sinr_db[ue_id] = float(np.mean(sinr_db_array))
            
            # Precompute sorted PRB indices (best to worst)
            sorted_prb_indices[ue_id] = np.argsort(-sinr_array)
        
        # --------------------------------------------------
        # IMPROVED: Better UE Clustering with More Flexible Boundaries
        # --------------------------------------------------
        
        # Sort UEs by average SINR
        sorted_ues = sorted(avg_sinr_db.items(), key=lambda x: x[1])
        
        # Create more adaptable SINR clusters with dynamic boundaries
        # Use percentile-based approach instead of fixed dB difference
        num_clusters = min(5, len(sorted_ues))  # Limit max clusters based on total UEs
        cluster_size = max(1, len(sorted_ues) // num_clusters)
        
        sinr_clusters = []
        for i in range(0, len(sorted_ues), cluster_size):
            cluster = [ue_id for ue_id, _ in sorted_ues[i:i+cluster_size]]
            if cluster:  # Only add non-empty clusters
                sinr_clusters.append(cluster)
        
        # --------------------------------------------------
        # OPTIMIZATION 3: Efficient PRB Tracking
        # --------------------------------------------------
        
        # Use boolean array for PRB availability
        available_prbs = np.ones(self.num_rbs, dtype=bool)
        
        # --------------------------------------------------
        # IMPROVED: Better Initial Allocation Strategy
        # --------------------------------------------------
        
        # First ensure every UE gets at least one PRB to avoid starvation
        # Prioritize UEs with poorer conditions for fairer initial allocation
        
        # Start with UEs in clusters from worst to best SINR
        for cluster in sinr_clusters:
            for ue_id in cluster:
                # Skip if already allocated in a previous pass
                if self.rb_allocation[ue_id]:
                    continue
                    
                # Find best available PRB for this UE
                for prb_idx in sorted_prb_indices[ue_id]:
                    if available_prbs[prb_idx]:
                        # Allocate this PRB
                        self.rb_allocation[ue_id].append(prb_idx)
                        available_prbs[prb_idx] = False
                        delivered_bits[ue_id] += rate_map[ue_id][prb_idx]
                        break  # One PRB per UE in this phase
        
        # --------------------------------------------------
        # IMPROVED: Proportional Fairness Allocation with Demand Awareness
        # --------------------------------------------------
        
        # Calculate normalized demand for each UE (for demand-aware allocation)
        max_demand = max(ue.demand for _, ue in self.ues.items())
        normalized_demand = {ue_id: self.ues[ue_id].demand / max_demand 
                            for ue_id in self.ues}
        
        # Calculate how many PRBs are still available
        remaining_prbs_count = np.sum(available_prbs)
        
        # If PRBs remain, allocate them using improved PF with demand awareness
        if remaining_prbs_count > 0:
            # Get remaining PRB indices
            remaining_prb_indices = np.where(available_prbs)[0]
            
            # Allocate remaining PRBs
            for prb in remaining_prb_indices:
                best_metric = -float('inf')
                best_ue = None
                
                for ue_id in self.ues:
                    # Skip if demand is already satisfied
                    current_rate = delivered_bits[ue_id]
                    max_demand_bits = self.ues[ue_id].demand * 1e6
                    
                    if current_rate >= max_demand_bits:
                        continue
                    
                    # Avoid division by zero with a safe minimum EWMA
                    ewma = max(self.ues[ue_id].ewma_dr, 1e-6)
                    
                    # Improved metric calculation:
                    # - Higher weight for UEs far from their demand
                    # - Consider both instantaneous rate and long-term fairness
                    demand_satisfaction = current_rate / max_demand_bits if max_demand_bits > 0 else 1.0
                    demand_weight = 1.0 / (0.1 + 0.9 * demand_satisfaction)  # Higher weight for unsatisfied demand
                    
                    # Combine rate, fairness, and demand factors
                    metric = (rate_map[ue_id][prb] / ewma) * demand_weight
                    
                    if metric > best_metric:
                        best_metric = metric
                        best_ue = ue_id
                
                # Allocate PRB to best UE if found
                if best_ue is not None:
                    self.rb_allocation[best_ue].append(prb)
                    available_prbs[prb] = False
                    delivered_bits[best_ue] += rate_map[best_ue][prb]
        
        # --------------------------------------------------
        # IMPROVED: Final Rebalancing for Zero-Allocation UEs
        # --------------------------------------------------
        
        # Check for UEs with zero PRBs - more aggressive redistribution
        zero_prb_ues = [ue_id for ue_id, prbs in self.rb_allocation.items() if not prbs]
        
        if zero_prb_ues:
            # Find UEs with multiple PRBs as potential donors
            ue_prb_counts = [(ue_id, len(prbs), delivered_bits[ue_id] / (self.ues[ue_id].demand * 1e6)) 
                            for ue_id, prbs in self.rb_allocation.items() if len(prbs) > 1]
            
            # Sort donors by satisfaction ratio (most satisfied first)
            ue_prb_counts.sort(key=lambda x: x[2], reverse=True)
            
            # Try to help each zero-PRB UE
            for zero_ue in zero_prb_ues:
                # Find best donor with multiple PRBs
                for donor_idx, (donor_ue, prb_count, _) in enumerate(ue_prb_counts):
                    if prb_count <= 1:  # Don't take from UEs with only 1 PRB
                        continue
                    
                    # Take the worst PRB from donor (least impact to donor)
                    donor_prbs = self.rb_allocation[donor_ue]
                    donor_rates = [rate_map[donor_ue][prb] for prb in donor_prbs]
                    worst_idx = np.argmin(donor_rates)
                    prb = donor_prbs[worst_idx]
                    
                    # Only donate if it helps zero_ue more than it hurts donor_ue
                    if rate_map[zero_ue][prb] > rate_map[donor_ue][prb] * 0.8:  # 20% efficiency threshold
                        # Transfer the PRB
                        donor_prbs.pop(worst_idx)
                        self.rb_allocation[zero_ue].append(prb)
                        
                        # Update delivered bits
                        delivered_bits[zero_ue] += rate_map[zero_ue][prb]
                        delivered_bits[donor_ue] -= rate_map[donor_ue][prb]
                        
                        # Update donor's PRB count
                        ue_prb_counts[donor_idx] = (donor_ue, prb_count - 1, 
                                                delivered_bits[donor_ue] / (self.ues[donor_ue].demand * 1e6))
                        ue_prb_counts.sort(key=lambda x: x[2], reverse=True)
                        break
        
        # --------------------------------------------------
        # Final EWMA Update and Resource Calculation
        # --------------------------------------------------
        
        # Update EWMA throughput for each UE
        for ue_id, ue in self.ues.items():
            ue.update_ewma(delivered_bits[ue_id])
            if ue.ewma_dr < 1e-6:  # Set a minimal EWMA
                ue.ewma_dr = 1e-6
        
        # Calculate allocated resources in bps
        self.allocated_resources = {}
        for ue_id, prbs in self.rb_allocation.items():
            if prbs:  # Only calculate for UEs with allocated PRBs
                self.allocated_resources[ue_id] = sum(rate_map[ue_id][prb] for prb in prbs)
            else:
                self.allocated_resources[ue_id] = 0.0
        
        # # Display allocation results
        # for ue_id, ue in self.ues.items():
        #     alloc_mbps = self.allocated_resources.get(ue_id, 0.0) / 1e6
        #     print(f" BS {self.reuse_color},UE {ue_id}: allocated={alloc_mbps:.2f} Mbps, demand={ue.demand:.2f} Mbps")

        # Calculate system load
        self.calculate_load()
        

    def calculate_capacity_rb_based(self, sample_points=500, overhead_factor=0.8):
        total_se = 0.0
        
        # Debug for macro
        is_macro = self.reuse_color == "Macro"
        # if self.id == 0:
        #     print(f"--- BS {self.id} Capacity Calculation ---")
        #     print(f"Frequency: {self.frequency/1e6:.1f} MHz")
        #     print(f"Bandwidth: {self.bandwidth/1e6:.1f} MHz")
        #     print(f"Num RBs: {self.num_rbs}")
            # print(f"TX Power: {self.tx_power_dbm} dBm")
        
        # 1) Determine a coverage radius - FIXED APPROACH
        # For macro: standard coverage radius
        # For small cells: either use distances between cells or a default
        if is_macro:
            cell_radius = 500.0  # More realistic macro coverage radius
        else:
            # For small cells, use distance to nearest neighbor or default
            same_tier_bs = [bs for bs in self.base_stations 
                        if bs.id != self.id and bs.reuse_color == self.reuse_color]
            
            if same_tier_bs:
                cell_radius = min(
                    np.linalg.norm(bs.position - self.position)
                    for bs in same_tier_bs
                ) / 2.0  # Half the distance to nearest same-tier BS
            else:
                cell_radius = 100.0  # Default small cell radius
        
        # 2) Cap instantaneous SINR at 30 dB (≈1000 linear)
        max_sinr_db = 30.0
        sinr_cap = 10 ** (max_sinr_db / 10)
        
        # Track average SINR for debugging
        sinr_samples = []
        
        # 3) Uniform‐disk sampling
        for _ in range(sample_points):
            u = np.random.rand()          # uniform [0,1)
            r = np.sqrt(u) * cell_radius  # uniform area
            theta = np.random.rand() * 2 * np.pi
            dx, dy = r * np.cos(theta), r * np.sin(theta)
            sample_point = self.position + np.array([dx, dy], dtype=np.float32)
            
            # 4) Compute desired & cross‐tier interference + noise
            prx = self.received_power_mW(sample_point)
            
            # Consider different interference models for macro vs small cells
            if is_macro:
                # For macro, small cells generally operate in different frequency bands
                # so they cause minimal interference
                interf = sum(
                    other.received_power_mW(sample_point) * 0.01  # Reduced cross-tier interference
                    for other in self.base_stations
                    if other.id != self.id and other.reuse_color != "Macro"  # Only from other tiers
                )
            else:
                # Small cells see interference from same color cells and reduced from macro
                interf = sum(
                    other.received_power_mW(sample_point) * (1.0 if other.reuse_color == self.reuse_color else 0.1)
                    for other in self.base_stations
                    if other.id != self.id
                )
                
            noise = self.noise_mW()
            
            sinr = min(prx / (interf + noise + 1e-12), sinr_cap)
            sinr_samples.append(sinr)
            total_se += np.log2(1 + sinr)
        
        # 5) From average spectral efficiency to capacity
        avg_se = total_se / sample_points       # bits/s/Hz
        rb_cap = self.rb_bandwidth * avg_se     # bits/s per RB
        total_bps = rb_cap * self.num_rbs * overhead_factor
        
        # Safety minimum capacities based on technology
        min_capacity = 100.0 if is_macro else 50.0
        
        # Mbps, with a guardrail at 10 Gbps and minimum capacity
        self.capacity = max(min(total_bps / 1e6, 1e4), min_capacity)
        
        # # Additional debug for macro
        # if self.id == 0:
        #     avg_sinr = sum(sinr_samples) / len(sinr_samples)
        #     print(f"Cell radius: {cell_radius:.1f} m")
        #     print(f"Avg SINR: {10*np.log10(avg_sinr):.2f} dB")
        #     print(f"Avg SE: {avg_se:.4f} bits/s/Hz")
        #     print(f"RB capacity: {rb_cap/1e6:.4f} Mbps")
        #     print(f"Total theoretical: {total_bps/1e6:.4f} Mbps")
        #     print(f"Final capacity: {self.capacity:.4f} Mbps")
        #     print("-----------------------------------")
        
        return self.capacity
    # def calculate_capacity_rb_based(self, sample_points=500, overhead_factor=0.8):
    #     """Use tier-appropriate capacity models"""
    #     if self.reuse_color == "Macro":
    #         # For macro: statistical model based on typical macro cell performance
    #         # This adjusts for the lack of neighboring macro cells
            
    #         # 1) Area: π·r² where r is effective coverage radius
    #         coverage_radius = 750.0  # meters (typical half-ISD)
    #         coverage_area = np.pi * coverage_radius**2  # m²
            
    #         # 2) Average spectral efficiency from empirical studies
    #         # Typically 1.5-3 bps/Hz for macro cells with modern technology
    #         avg_spectral_efficiency = 2.5  # bps/Hz
            
    #         # 3) Apply bandwidth and overhead factors
    #         total_bps = avg_spectral_efficiency * self.bandwidth * overhead_factor
            
    #         # 4) Cell capacity (bps)
    #         self.capacity = total_bps / 1e6  # Convert to Mbps
    #     else:
    #         # For small cells: use your existing detailed calculation
    #         total_se = 0.0
    #         # 1) Determine a coverage radius
    #         if len(self.base_stations) == 1:
    #             # fallback radii: macro vs. small cell
    #             cell_radius = 1000.0 if self.reuse_color == "Macro" else 100.0
    #         else:
    #             cell_radius = max(
    #                     np.linalg.norm(bs.position - self.position)
    #                     for bs in self.base_stations
    #                 )

    #             # 2) Cap instantaneous SINR at 30 dB (≈1000 linear)
    #         max_sinr_db = 30.0
    #         sinr_cap    = 10 ** (max_sinr_db / 10)

    #             # 3) Uniform‐disk sampling
    #         for _ in range(sample_points):
    #             u     = np.random.rand()          # uniform [0,1)
    #             r     = np.sqrt(u) * cell_radius  # uniform area
    #             theta = np.random.rand() * 2 * np.pi
    #             dx, dy = r * np.cos(theta), r * np.sin(theta)
    #             sample_point = self.position + np.array([dx, dy], dtype=np.float32)

    #             # 4) Compute desired & cross‐tier interference + noise
    #             prx   = self.received_power_mW(sample_point)
    #             interf = sum(
    #                     other.received_power_mW(sample_point)
    #                     for other in self.base_stations
    #                     if other.id != self.id
    #                 )
    #             noise = self.noise_mW()

    #             sinr = min(prx / (interf + noise + 1e-12), sinr_cap)
    #             total_se += np.log2(1 + sinr)

    #             # 5) From average spectral efficiency to capacity
    #             avg_se    = total_se / sample_points       # bits/s/Hz
    #             rb_cap    = self.rb_bandwidth * avg_se     # bits/s per RB
    #             total_bps = rb_cap * self.num_rbs * overhead_factor

    #             # Mbps, with a guardrail at 10 Gbps (for modest BW)
    #             self.capacity = min(total_bps / 1e6, 1e4)
    #             return self.capacity
                
    def snr_per_rb(self, ue):
        sinr_rb = np.empty(self.num_rbs, dtype=np.float32)
        noise_rb = self.noise_mW()
        
        # Calculate base received power
        base_prx = self.received_power_mW(ue.position, ue.rx_gain)
        
        # Generate frequency-selective fading per RB (simplified model)
        # Correlated Rayleigh fading with correlation across adjacent RBs
        coherence_rbs = min(20, self.num_rbs // 50)  # Coherence bandwidth in RBs
        num_independent_fades = max(1, self.num_rbs // coherence_rbs)
        
        # Generate independent fades
        independent_fades = np.random.rayleigh(scale=1.0, size=num_independent_fades)
        
        # Interpolate to get per-RB fading
        fading = np.interp(
            np.linspace(0, num_independent_fades-1, self.num_rbs),
            np.arange(num_independent_fades),
            independent_fades
        )
        
        # Normalize fading to maintain average power
        fading = fading / np.mean(fading)
        
        # Apply fading to received power per RB
        prx_per_rb = base_prx * fading
        
        # Calculate interference (could also add per-RB interference variation)
        interf = 0.0
        for other in self.base_stations:
            if other.id == self.id or other.reuse_color != self.reuse_color:
                continue
            interf += other.received_power_mW(ue.position, ue.rx_gain)
        
        # Calculate SINR per RB
        for rb in range(self.num_rbs):
            sinr_rb[rb] = prx_per_rb[rb] / (interf + noise_rb + 1e-12)
        
        return sinr_rb
    
    
class NetworkEnvironment(MultiAgentEnv):
    # @staticmethod
    # def generate_hex_positions(
    #     num_bs,
    #     width=100.0,
    #     height=100.0,
    #     min_distance_from_center=30.0,  # Minimum distance for small cells
    #     enforce_center=True
    # ):
    #     """
    #     Generates positions with:
    #     - Guaranteed central BS at (50,50) when enforce_center=True
    #     - Other BSs placed in hex pattern ≥ min_distance_from_center away
    #     """
    #     center = (width/2, height/2)
    #     positions = []
        
    #     if num_bs < 1:
    #         return []

    #     # ==================================================================
    #     # 1. Always place first BS at center (50,50)
    #     # ==================================================================
    #     if enforce_center:
    #         positions.append(center)
    #         remaining_bs = num_bs - 1
    #     else:
    #         remaining_bs = num_bs

    #     # ==================================================================
    #     # 2. Calculate hexagonal grid parameters for remaining BSs
    #     # ==================================================================
    #     if remaining_bs > 0:
    #         # Effective area excluding central safety zone
    #         safe_radius = min_distance_from_center
    #         usable_width = width - 2*safe_radius
    #         usable_height = height - 2*safe_radius
            
    #         area_per_bs = (usable_width * usable_height) / remaining_bs
    #         pitch = max(math.sqrt(2 * area_per_bs / math.sqrt(3)), 15.0)
            
    #         # Generate grid positions offset from center
    #         hex_height = pitch * math.sin(math.radians(60))
    #         n_cols = int(math.ceil(usable_width / pitch)) + 2
    #         n_rows = int(math.ceil(usable_height / hex_height)) + 2
            
    #         x_start = safe_radius
    #         y_start = safe_radius
            
    #         # ==============================================================
    #         # 3. Generate candidate positions (all outside safe radius)
    #         # ==============================================================
    #         candidate_pos = []
    #         for row in range(n_rows):
    #             y = y_start + row * hex_height
    #             x_offset = (pitch/2 if row % 2 else 0)
                
    #             for col in range(n_cols):
    #                 x = x_start + x_offset + col * pitch
    #                 candidate = (x, y)
                    
    #                 # Check position validity
    #                 if (0 <= x <= width) and (0 <= y <= height):
    #                     dist_to_center = np.hypot(x-center[0], y-center[1])
    #                     if dist_to_center >= min_distance_from_center:
    #                         candidate_pos.append(candidate)

    #         # ==============================================================
    #         # 4. Sort candidates by distance from center (spiral pattern)
    #         # ==============================================================
    #         candidate_pos.sort(
    #             key=lambda p: (
    #                 np.hypot(p[0]-center[0], p[1]-center[1]),  # Distance
    #                 -np.arctan2(p[1]-center[1], p[0]-center[0]) # Angle
    #             )
    #         )

    #         # Take first N positions meeting criteria
    #         positions += candidate_pos[:remaining_bs]

    #     return positions[:num_bs]
    @staticmethod
    def generate_hex_positions(num_bs, width=100.0, height=100.0, min_distance_from_center=30.0,  # Minimum distance for small cells
    enforce_center=True):
        """
        Generate base station positions with the macro cell at center (width/2, height/2)
        and small cells arranged in optimal positions around it.
        """
        if num_bs < 1:
            return []
        
        # Always place the first base station (macro cell) at the center
        center = (width/2, height/2)
        positions = [center]
        
        if num_bs == 1:
            return positions
        
        # For small cells, use optimal positioning instead of standard hexagonal grid
        # Calculate the effective radius - half of the smaller dimension
        effective_radius = min(width, height) / 2 * 0.65  # 65% of half-width for optimal small cell placement
        
        # Generate positions on concentric rings around the center
        remaining_positions = []
        
        # First ring - optimal for up to 6 small cells
        if num_bs <= 7:  # 1 macro + 6 small cells
            angle_step = 2 * math.pi / (num_bs - 1)
            for i in range(num_bs - 1):
                angle = i * angle_step
                x = center[0] + effective_radius * math.cos(angle)
                y = center[1] + effective_radius * math.sin(angle)
                remaining_positions.append((x, y))
        else:
            # First ring - 6 cells
            for i in range(6):
                angle = i * math.pi / 3
                x = center[0] + effective_radius * math.cos(angle)
                y = center[1] + effective_radius * math.sin(angle)
                remaining_positions.append((x, y))
            
            # If more cells are needed, add additional rings with increasing radius
            remaining_cells = num_bs - 7  # -1 for macro, -6 for first ring
            if remaining_cells > 0:
                # Second ring
                second_ring_radius = effective_radius * 1.8
                cells_in_second_ring = min(12, remaining_cells)
                angle_step = 2 * math.pi / cells_in_second_ring
                
                for i in range(cells_in_second_ring):
                    angle = i * angle_step + (angle_step / 2)  # offset to stagger from first ring
                    x = center[0] + second_ring_radius * math.cos(angle)
                    y = center[1] + second_ring_radius * math.sin(angle)
                    remaining_positions.append((x, y))
                
                # If still more cells needed, fall back to hexagonal grid for the rest
                remaining_cells -= cells_in_second_ring
                if remaining_cells > 0:
                    # Calculate standard hexagonal grid positions as before
                    area_per_bs = (width * height) / num_bs
                    pitch = math.sqrt((2 * area_per_bs) / math.sqrt(3))
                    pitch = min(pitch, min(width, height) / 2)
                    pitch = max(pitch, 5.0)
                    
                    hex_height = pitch * math.sin(math.radians(60))
                    n_cols = int(math.ceil(width / pitch)) + 2
                    n_rows = int(math.ceil(height / hex_height)) + 2
                    
                    x_offset = (width - (n_cols-1) * pitch) / 2
                    y_offset = (height - (n_rows-1) * hex_height) / 2
                    
                    # Generate grid positions
                    grid_positions = []
                    for row in range(n_rows):
                        y = y_offset + row * hex_height
                        x_start = x_offset + (pitch/2 if row % 2 else 0)
                        
                        for col in range(n_cols):
                            x = x_start + col * pitch
                            pos = (x, y)
                            # Skip the center and any positions too close to already placed cells
                            if 0 <= x <= width and 0 <= y <= height and pos != center:
                                min_distance = min([math.hypot(x-p[0], y-p[1]) for p in positions + remaining_positions], default=float('inf'))
                                if min_distance > pitch * 0.7:
                                    grid_positions.append(pos)
                    
                    # Sort remaining grid positions by distance from center
                    grid_positions.sort(key=lambda p: math.hypot(p[0]-center[0], p[1]-center[1]))
                    
                    # Add as many as needed
                    remaining_positions.extend(grid_positions[:remaining_cells])
        
        # Ensure all positions are within bounds
        remaining_positions = [(max(0, min(width, x)), max(0, min(height, y))) for x, y in remaining_positions]
        
        # Return the center followed by the optimally placed small cells
        return positions + remaining_positions[:num_bs-1]
    
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
        self.seed = 42
        if self.seed is not None:
            np.random.seed(self.seed)
            
        self.step_count       = 0        
        self.base_stations = []
        # Calculating hexagonal positions for BS             
        
        # In NetworkEnvironment.__init__:
        grid = self.generate_hex_positions(
            num_bs=self.num_bs,
            min_distance_from_center=30.0,
            enforce_center=True
        )
        # In NetworkEnvironment __init__ after grid generation:
        # In NetworkEnvironment __init__ after grid generation:
        colors = ["A", "B", "C"]  # Simple 3-color reuse
        # Define bandwidths (aligned with 3GPP recommendations)
        MACRO_CONFIG = {
            "frequency": 2.1e9,    # Sub-6 GHz
            "bandwidth": 20e6,     # 20 MHz (typical macro cell)
            "tx_power_dbm": 46.0,
            "path_loss_n": 3.5
        }

        SMALL_CELL_CONFIG = {
            "A": {
                "frequency": 26.75e9,  # # full n258 BW 28 GHz
                "bandwidth": 3.25e9    # 2 GHz (mmWave typical)
            },
            "B": {
                "frequency": 28.00e9,  # # full n257 BW 39 GHz
                "bandwidth": 3.00e9  # 4 GHz 
            },
            "C": {
                "frequency": 39.00e9,  #  # full n260 BW 60 GHz
                "bandwidth": 3.00e9   # 8 GHz
            }
        }

        # Macro BS initialization
        macro_bs = BaseStation(
            id=0,
            position=grid[0],
            frequency=MACRO_CONFIG["frequency"],
            bandwidth=MACRO_CONFIG["bandwidth"],
            subcarrier_spacing=15e3,
            reuse_color="Macro",
            tx_power_dbm=MACRO_CONFIG["tx_power_dbm"],
            path_loss_n=MACRO_CONFIG["path_loss_n"],
            tx_gain_dbi=8.0,    # Standard cellular antenna
            bf_gain_dbi=10.0,    # Limited beamforming
            path_loss_sigma=8.0,
            cre_bias=0.0            
        )
        self.base_stations.append(macro_bs)
        # Small cells initialization
        small_cells = []
        for i, pos in enumerate(grid[1:]):  # Skip macro
            color = colors[i % len(colors)]
            config = SMALL_CELL_CONFIG[color]
            
            small_cells.append(BaseStation(
                id=i+1,
                position=pos,
                frequency=config["frequency"],
                bandwidth=config["bandwidth"],
                subcarrier_spacing=60e3,
                reuse_color=color,
                tx_power_dbm=30.0,
                path_loss_n=2.1,
                path_loss_sigma=4.0,
                cre_bias=6.0,
                tx_gain_dbi=12.0,  # High-gain mmWave antenna
                bf_gain_dbi=25.0,   # Advanced beamforming
                height=10.0         # Small cell height
            ))
        self.base_stations.extend(small_cells)
        all_positions = np.random.uniform(0, 100, size=(self.num_ue, 2)).astype(np.float32)
        self.ues = [
            UE(
                id=i,
                position=all_positions[i],
                demand=np.random.randint(5, 20),
                v_min=0.5,
                v_max=1.5,
                pause_min=1.0,
                pause_max=5.0,
                ewma_alpha=0.9
            )
            for i in range(self.num_ue)
        ]
        print(f"Created {len(self.ues)} UEs")
        # self.ue_positions = {}
        # for ue in self.ues:
        #     agent_id = f"ue_{ue.id}"  # Convert int ID to string format
        #     self.ue_positions[agent_id] = ue.position
        self.prev_associations = {ue.id: None for ue in self.ues}
        self.handover_counts  = {ue.id: 0    for ue in self.ues}
        self.load_history = {bs.id: [] for bs in self.base_stations}
        self.associations = {bs.id: [] for bs in self.base_stations}
        self.initial_assoc   = config.get("initial_assoc", None)
        self._has_warm_start = False
        for bs in self.base_stations:
            bs.base_stations = self.base_stations  # for interference loops
            bs.ues           = self.ues            # so data_rate_shared can see all UEs
        # Initialize KPI logger if logging is enabled
        self.kpi_logger = KPITracker() if log_kpis else None        
        # obs_dim = 3*self.num_bs + 1 + (self.num_bs + 1) # SINRs + BS loads + BS Utilizations + own demand + Connected
        obs_dim = 3*self.num_bs + 3 + (self.num_bs + 1)
        self.observation_space = gym.spaces.Dict({
            f"ue_{i}": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_dim,), dtype=np.float32
            )
            for i in range(self.num_ue)
        })


        self.action_space = gym.spaces.Dict({
            f"ue_{i}": gym.spaces.Discrete(self.num_bs + 1)
            for i in range(self.num_ue)
        })

        self._initialize_policy_manager()
        
    def _initialize_policy_manager(self):
        """Initialize the policy mapping manager with current BS and UE positions"""
        # Extract BS positions
        bs_positions = np.array([bs.position for bs in self.base_stations])
        
        # Extract initial UE positions
        initial_ue_positions = {}
        for ue in self.ues:
            agent_id = f"ue_{ue.id}"
            initial_ue_positions[agent_id] = np.array(ue.position)
        
        # Create policy manager
        self.policy_manager = PolicyMappingManager(bs_positions, initial_ue_positions)
        
        print(f"Policy manager initialized with {len(bs_positions)} BSs and {len(initial_ue_positions)} UEs")
        print("Initial policy distribution:")
        self.policy_manager.log_policy_assignments()
        
    def get_policy_for_agent(self, agent_id: str) -> str:
        """Get the policy name for a given agent based on current position"""
        closest_bs = self.policy_manager.get_closest_bs(agent_id)
        return f"bs_{closest_bs}_policy"
    
    def get_policy_distribution(self) -> Dict[str, int]:
        """Get current policy distribution across all UEs"""
        return self.policy_manager.get_policy_distribution()
    
    def log_policy_status(self):
        """Log current policy assignments - useful for debugging"""
        return self.policy_manager.log_policy_assignments()
    
    def reward(self, agent):
        return self.calculate_individual_reward(agent)  # Implement per-BS reward

    def calculate_individual_reward(self, agent_id=None):
        if agent_id is None:
            return 0.0

        # Parse UE index
        if isinstance(agent_id, str) and agent_id.startswith("ue_"):
            ue_id = int(agent_id.split("_")[1])
            ue = self.ues[ue_id]

            # 1) Unconnected penalty
            if ue.associated_bs is None:
                return -1.0

            # 2) SINR factor (linear → clipped & normalized)
            lin_snr = ue.sinr  # already linear
            # Clip to [0, SNR_max] and normalize
            SNR_MAX = 100.0
            snr_clipped = max(0.0, min(lin_snr, SNR_MAX))
            sinr_factor = snr_clipped / SNR_MAX  # ∈ [0,1]

            # 3) Load factor (PRB utilization)
            bs = self.base_stations[ue.associated_bs]
            prb_util = bs.load / (bs.num_rbs + 1e-9)  # ∈ [0,1]
            load_factor = 1.0 - prb_util           # ∈ [0,1]

            # 4) Composite reward
            #    weight SINR high if you care throughput, weight load high if you care fairness
            w_snr, w_load = 0.7, 0.3
            base_reward = w_snr * sinr_factor + w_load * load_factor

            # 5) Scale into a convenient range (e.g. [–1, +1])
            #    Here we map base_reward ∈ [0,1] → [0,+1], then shift down for unconnected
            return float(base_reward)

        return 0.0
    
    def reset(self, *, seed=None, options=None):
        # 1) Reset step counter and clear all BS loads & UE state
        self.current_step = 0
        for bs in self.base_stations:
            bs.allocated_resources.clear()
            bs.calculate_load()
        for ue in self.ues:
            ue.associated_bs = None
            ue.sinr          = -np.inf
            ue.ewma_dr       = 0.0

        # 2) ONE-TIME metaheuristic warm start
        if not self._has_warm_start and self.initial_assoc is not None:
            for ue in self.ues:
                bs_idx = self.initial_assoc[ue.id]
                ue.associated_bs = bs_idx
                dr = self.base_stations[bs_idx].data_rate_shared(ue)
                self.base_stations[bs_idx].allocated_resources[ue.id] = dr
            # Recompute each BS’s load once after all assignments
            for bs in self.base_stations:
                bs.calculate_load()

            # Mark that warm start has been applied
            self._has_warm_start = True
            # Optionally clear to free memory
            self.initial_assoc = None
            # Reinitialize policy manager with reset positions
        self._initialize_policy_manager()
        
        # 3) Build and return the obs + infos dicts
        obs   = self._get_obs()  
        infos = {f"ue_{i}": {} for i in range(self.num_ue)}
        return obs, infos
        # return self._get_obs()# , {}

    # Add these to your NetworkEnvironment class
    def calculate_jains_fairness(self):
        throughputs = [ue.throughput for ue in self.ues]
        return (sum(throughputs) ** 2) / (len(throughputs) * sum(t**2 for t in throughputs))

    @property
    def throughput(self):
        return torch.log2(1 + 10**(self.sinr/10)).item()
        
    def calculate_reward(self):
        """
        Normalized reward: Gbps throughput + fairness - overload_penalty.
        
        Assumptions:
          - Each BS stores per-UE throughput in bits/s in bs.allocated_resources (dict ue_id->bits/s).
          - bs.capacity is in Mbps (set by calculate_capacity_rb_based()).
        """
        # 1) Total system throughput (bits/s) → Gbps
        total_bps = sum(
            dr
            for bs in self.base_stations
            for dr in bs.allocated_resources.values()
        )
        throughput_gbps = total_bps / 1e9

        # 2) Refresh each BS capacity (Mbps)
        for bs in self.base_stations:
            bs.capacity = bs.calculate_capacity_rb_based()
            # print(f"BS {bs.id}, Capacity: {bs.capacity}")
        # 3) Build load and capacity tensors (both in bps)
        loads_bps = torch.tensor(
            [sum(bs.allocated_resources.values()) for bs in self.base_stations],
            dtype=torch.float32
        )
        capacities_bps = torch.tensor(
            [bs.capacity * 1e6 for bs in self.base_stations],
            dtype=torch.float32
        )

        # 4) Normalized load per BS (unitless)
        loads_norm = loads_bps / (capacities_bps + 1e-9)

        # 5) Jain’s fairness index on normalized loads
        #    J = (sum x_i)^2 / (N * sum x_i^2)
        N = len(self.base_stations)
        sum_loads = loads_norm.sum()
        fairness = (sum_loads * sum_loads) / (N * (loads_norm * loads_norm).sum() + 1e-6)

        # 6) Overload penalty: sum of (load_norm - 1)+ across BSs
        overload = torch.relu(loads_norm - 1.0).sum()
        # for i, bs in enumerate(self.base_stations):
        #     load = sum(bs.allocated_resources.values())
        #     capacity = bs.capacity * 1e6
        #     print(f"BS {bs.id}: Load={load/1e6:.2f}Mbps, Capacity={capacity/1e6:.2f}Mbps, Ratio={load/capacity:.2f}")
        # 7) Composite reward
        #    - throughput in Gbps (≈ 0–X)
        #    - fairness [0–1]
        #    - overload penalty (unitless)
        reward = (
            1.0 * throughput_gbps   # reward raw capacity in Gbps
            + 2.0 * fairness        # weight fairness
            - 1.0 * overload        # penalize overloaded cells
        )
        return reward

    
    def pick_best_bs(self, ue, w1=0.4, w2=0.3, w3=0.2, w4=0.1, delta=0.1):
        # 1) Instantaneous PF-style rate on the best PRB for each BS
        inst_rates = []
        for bs in self.base_stations:
            # get per-RB SINR array for this UE
            sinr_rb = bs.snr_per_rb(ue)                     # linear
            # best PRB rate = max_over_p(RB_bw * log2(1+SINR_p))
            best_rate = np.max(self.rb_bandwidth * np.log2(1 + sinr_rb))
            inst_rates.append(best_rate / ue.ewma_dr)       # PF metric
        inst_rates = np.array(inst_rates)

        # 2) Load-based free-RB score
        load_scores = np.array([
            1 - (sum(len(r) for r in bs.rb_allocation.values()) / bs.num_rbs)
            for bs in self.base_stations
        ])

        # 3) Normalized SINR across BSs
        sinrs = self._calculate_sinrs(ue)                  # returns linear array per BS
        sinr_norm = sinrs / (sinrs.max() + 1e-6)

        # 4) Composite score
        scores = (
            w1 * inst_rates +
            w2 * load_scores +
            w3 * (inst_rates / (inst_rates.max() + 1e-6)) +
            w4 * sinr_norm
        )

        # 5) Hysteretic selection
        cand = int(np.argmax(scores))
        if ue.associated_bs is not None:
            old = ue.associated_bs
            if scores[cand] >= scores[old] + delta:
                return cand
            else:
                return old
        return cand



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
        
        
    # def _calculate_sinrs(self, ue):
    #     sinrs = []
    #     for bs in self.base_stations:
    #         signal = bs.received_power_mW(ue.position)
    #         # use the method you wrote to compute interference at the BS—
    #         # here we just compute interference *at the UE*, so we inline it
    #         interf = sum(
    #         other.received_power_mW(ue.position)
    #         for other in self.base_stations
    #         if (other.reuse_color == bs.reuse_color) and (other.id != bs.id)
    #         )
    #         noise = bs.noise_mW()
    #         sinrs.append(10 * np.log10(signal/(interf+noise+1e-12)))
            
    #     return np.array(sinrs)
    
    def _calculate_sinrs(self, ue, debug=False):
        """
        Return an array of linear SINRs from every BS → UE link.
        If debug=True, also print the first few link budgets in dB.
        """
        sinrs = []
        for bs in self.base_stations:
            # 1) desired signal power (mW)
            prx = bs.received_power_mW(ue.position, ue.rx_gain)

            # 2) co-channel interference (mW)
            interf = sum(
                other.received_power_mW(ue.position, ue.rx_gain)
                for other in self.base_stations
                if (other.id != bs.id) and (other.reuse_color == bs.reuse_color)
            )

            # 3) noise power (mW)
            noise = bs.noise_mW()

            # 4) linear SINR
            lin_snr = prx / (interf + noise + 1e-12)

            # Optional debug print in dB
            if debug and ue.id < 3:
                snr_db = 10 * np.log10(lin_snr)
                print(f" UE {ue.id} → BS {bs.id}: "
                    f"S={prx:.3e} mW, I={interf:.3e} mW, N={noise:.3e} mW, "
                    f"SINR={snr_db:.2f} dB")

            sinrs.append(lin_snr)

        return np.array(sinrs)


    
    def _update_sinrs(self):
        for ue in self.ues:
            if ue.associated_bs is not None:
                sinrs = self._calculate_sinrs(ue)
                ue.sinr = sinrs[ue.associated_bs]
            else:
                ue.sinr = -np.inf
    def step(self, actions):
        try:
            print(f"Step called with {len(actions)} actions")
            start_time = time.time()            
            connected_count = 0
            handover_count = 0
            # 1) Decode and apply associations
            for agent_id, a in actions.items():
                idx = int(agent_id.split("_")[1])
                self.ues[idx].associated_bs = None if a == 0 else (a-1)
                connected_count += 1

            # 2) Run PF scheduler on each BS
            for bs in self.base_stations:
                bs.ues = {ue.id: ue for ue in self.ues if ue.associated_bs == bs.id}
                bs.allocate_prbs()

            # 3) Update SINR & EWMA
            self._update_system_metrics()
            # print(f"Connected Users : {connected_count} Users")
            # 4) Compute per-agent rewards
            rewards = {f"ue_{ue.id}": self.calculate_individual_reward(f"ue_{ue.id}")
                    for ue in self.ues}
            step_time = time.time() - start_time
            # 4b) Compute aggregate metrics for logging
            total_reward = sum(rewards.values())
            # Connected ratio
            connected_ratio = sum(1 for ue in self.ues if ue.associated_bs is not None) / self.num_ue

            # Load‐balancing fairness (Jain) on normalized PRB loads
            loads = np.array([bs.load for bs in self.base_stations], dtype=np.float32)
            prb_caps = np.array([bs.num_rbs for bs in self.base_stations], dtype=np.float32)
            util = loads / (prb_caps + 1e-9)
            if util.sum() > 0:
                jains = (util.sum()**2) / (len(util) * (util**2).sum() + 1e-9)
            else:
                jains = 0.0

            # Current association solution
            current_solution = [
                ue.associated_bs if ue.associated_bs is not None else -1
                for ue in self.ues
            ]

            # SINR list (capped at 100 dB for safety)
            sinr_list = [
                float(min(ue.sinr, 100.0)) if ue.associated_bs is not None else -np.inf
                for ue in self.ues
            ]

            # Safe throughput sum (Gbps)
            total_throughput = 0.0
            for ue in self.ues:
                if ue.associated_bs is not None:
                    lin = min(ue.sinr, 100.0)
                    total_throughput += np.log2(1 + 10**(lin/10))
                    
            # now total_throughput is in bits/s per Hz—if you want Gbps, multiply by your PRB_bw and num_prbs:self.rb_bandwidth 
            total_throughput_gbps = total_throughput * 180e3 * len(self.base_stations[0].rb_allocation) / 1e9

            # 4c) Log to KPI
            if self.log_kpis and self.kpi_logger:
                metrics = {
                    "connected_ratio": connected_ratio,
                    "step_time": step_time,
                    "episode_reward_mean": total_reward / self.num_ue,
                    "fairness_index": jains,
                    "throughput_sum": total_throughput_gbps,
                    "solution": current_solution,
                    "sinr_list": sinr_list,
                }
                self.kpi_logger.log_metrics(
                    phase="environment",
                    algorithm="hybrid_marl",
                    metrics=metrics,
                    episode=self.current_step
                )
            # 5) Build infos, check termination
            obs = self._get_obs()
            terminated = {"__all__": False}
            truncated  = {"__all__": self.current_step >= self.episode_length}
            # 1) Compute per-UE instantaneous throughput
            per_agent_info = {}
            for ue in self.ues:
                key = f"ue_{ue.id}"
                connected = ue.associated_bs is not None
                sinr_val  = min(ue.sinr, 100.0) if connected else -np.inf
                thr       = np.log2(1 + 10**(sinr_val/10)) if connected else 0.0

                per_agent_info[key] = {
                    "connected":            connected,
                    "sinr_dB":              float(sinr_val),
                    "throughput_bps_per_hz": float(thr)
                }

            # 2) Compute global fairness (Jain’s index) on PRB utilization
            loads    = np.array([bs.load for bs in self.base_stations], dtype=np.float32)
            caps     = np.array([bs.num_rbs for bs in self.base_stations], dtype=np.float32) + 1e-9
            util     = loads / caps
            jains    = (util.sum()**2) / (len(util) * (util**2).sum() + 1e-9) if util.sum() > 0 else 0.0

            # 3) Assemble common (__all__) info
            common_info = {
                "connected_ratio":      connected_count / self.num_ue,
                "step_time_s":          step_time,
                "avg_reward":           total_reward / self.num_ue if self.num_ue > 0 else 0.0,
                "throughput_sum_Gbps":  total_throughput_gbps,
                "fairness_index":       float(jains),
                "current_solution":     current_solution
            }

            # Create info dict with one entry per agent, plus global info
            info = {
                f"ue_{ue.id}": {
                    "connected":             ue.associated_bs is not None,
                    "sinr_dB":               float(min(ue.sinr, 100.0)) if ue.associated_bs is not None else float("-inf"),
                    "throughput_bps_per_hz": float(
                        np.log2(1 + 10**(min(ue.sinr, 100.0) / 10))
                    ) if ue.associated_bs is not None else 0.0
                }
                for ue in self.ues
            }
            info["__common__"] = common_info
            self.current_step += 1
            # Update UE positions using MRWP mobility model
            for ue in self.ues:
                ue.update_position()
            
            # Update policy manager with new positions
            new_positions = {}
            for ue in self.ues:
                agent_id = f"ue{ue.id}"
                new_positions[agent_id] = np.array(ue.position)
            self.policy_manager.update_ue_positions(new_positions)
            
            return obs, rewards, terminated, truncated, info
        except Exception as e:        
            print(f"ERROR in step: {e}")
            import traceback
            print(traceback.format_exc())
            # Return a safe default response
            return self._get_obs(), {f"ue_{ue.id}": 0.0 for ue in self.ues}, {"__all__": False}, {"__all__": True}, {"__common__": {"error": str(e)}} 
    
    def get_last_info(self):
        """Return the last info dict from a step"""
        if hasattr(self, 'last_info'):
            print("Getting lastest info....")
            return self.last_info
        return None
    

    def _get_obs(self):
        """
        Get observation for each UE that includes:
        - Normalized SINR to each BS
        - PRB load fractions for each BS
        - BS utilization (rate/capacity ratio)
        - UE demand (normalized)
        - One-hot encoding of current association
        - Last-step throughput (bps/Hz) and global Jain fairness
        """
        # 1) PRB-based load fraction per BS
        bs_prb_loads = np.array([bs.load for bs in self.base_stations], dtype=np.float32)
        bs_num_rbs   = np.array([max(bs.num_rbs, 1) for bs in self.base_stations], dtype=np.float32)
        prb_fractions = bs_prb_loads / bs_num_rbs

        # 2) Rate-capacity utilization per BS
        rate_loads = np.array([sum(bs.allocated_resources.values()) for bs in self.base_stations],
                            dtype=np.float32)
        cap_bps    = np.array([max(bs.capacity * 1e6, 1.0) for bs in self.base_stations],
                            dtype=np.float32)
        util_bps   = np.clip(rate_loads / cap_bps, 0.0, 10.0)

        # 3) Compute global Jain's fairness index on PRB utilization
        util = prb_fractions  # or use util_bps/cap_bps if you prefer rate-based fairness
        if util.sum() > 0:
            global_jains = float((util.sum()**2) / (len(util) * (util**2).sum() + 1e-9))
        else:
            global_jains = 0.0

        obs = {}
        for ue in self.ues:
            # A) SINR vector normalized
            sinr_lin = self._calculate_sinrs(ue).astype(np.float32)
            max_sinr = max(sinr_lin.max(), 1e-6)
            norm_sinr = sinr_lin / max_sinr

            # B) Normalized demand
            ue_demand = max(ue.demand, 1.0)
            norm_demand = np.array([min(ue.demand / ue_demand, 1.0)], dtype=np.float32)

            # C) One-hot association
            idx = ue.associated_bs if ue.associated_bs is not None else self.num_bs
            one_hot = np.eye(self.num_bs + 1, dtype=np.float32)[idx]

            # D) Last-step per-UE throughput (bps/Hz)
            if ue.associated_bs is not None:
                sinr_dB = min(ue.sinr, 100.0)
                thr_bps_per_hz = np.log2(1 + 10**(sinr_dB / 10))
            else:
                thr_bps_per_hz = 0.0
            last_throughput = np.array([thr_bps_per_hz], dtype=np.float32)

            # E) Pack everything into one vector
            obs_vector = np.concatenate([
                norm_sinr,          # (num_bs,)
                prb_fractions,      # (num_bs,)
                util_bps,           # (num_bs,)
                norm_demand,        # (1,)
                one_hot,            # (num_bs+1,)
                last_throughput,    # (1,)
                np.array([global_jains], dtype=np.float32)  # (1,)
            ], axis=0)

            obs[f"ue_{ue.id}"] = obs_vector
            # obs[ue.id] = obs_vector

        return obs

   
    def _calculate_local_interference(self, neighbor_dist=100.0):
        interference = 0.0
        for other in self.base_stations:
            if other.id == self.id:
                continue
            # only count BSs on the same reuse_color
            if other.reuse_color != self.reuse_color:
                continue
            d = np.linalg.norm(self.position - other.position)
            if d < neighbor_dist:
                prx_int = other.received_power_mW(self.position)
                interference += prx_int
        return interference
   

    def _get_normalized_ue_positions(self, bs):
        """Normalize positions to [-1,1] range"""
        positions = []
        for ue in self.ues:
            rel_pos = (ue.position - bs.position).numpy()
            # Normalize based on environment bounds (0-100 in your case)
            norm_pos = rel_pos / 50.0 - 1.0  # Scales to [-1,1]
            positions.append(norm_pos.astype(np.float32))  # ✅ Cast
        return np.array(positions).flatten()
    
    def get_state_snapshot(self) -> dict:
         # pick the UE iterable based on type
            
            ue_iter = (self.ues.values() if isinstance(self.ues, dict)
                else self.ues)
            return {
            "users": [{
                "id": ue.id,
                "position": ue.position.copy().tolist(),
                "waypoint": ue.waypoint.copy().tolist(),
                "speed": float(ue.speed),
                "pause_time": float(ue.pause_time),
                "demand": float(ue.demand),
                "associated_bs": ue.associated_bs,
                "sinr": float(ue.sinr),
                "ewma_dr": float(ue.ewma_dr)
            } for ue in ue_iter],# self.ues
            "base_stations": [{
                "id": bs.id,
                "allocated_resources": bs.allocated_resources.copy(),
                "rb_allocation": {ue_id: prbs.copy() for ue_id, prbs in bs.rb_allocation.items()},
                "load": float(bs.load),
                "capacity": float(bs.capacity),
                "reuse_color": bs.reuse_color,
                "frequency": float(bs.frequency)
            } for bs in self.base_stations]
            # # Optional: histories & counters
            # "load_history": {bs.id: hist.copy() for bs, hist in self.load_history.items()},
            # "handover_counts": self.handover_counts.copy(),
            # "prev_associations": self.prev_associations.copy(),
            # "step_count": self.step_count,
            # "current_step": self.current_step
        }

    def set_state_snapshot(self, state: dict):
        # restore UEs
        ue_iter = self.ues.values() if isinstance(self.ues, dict) else self.ues
        for ue_state in state["users"]:
            ue = next(u for u in ue_iter if u.id == ue_state["id"])
            ue.position       = np.array(ue_state["position"], dtype=np.float32)
            ue.waypoint       = np.array(ue_state["waypoint"], dtype=np.float32)
            ue.speed          = ue_state["speed"]
            ue.pause_time     = ue_state["pause_time"]
            ue.demand         = ue_state["demand"]
            ue.associated_bs  = ue_state["associated_bs"]
            ue.sinr           = ue_state["sinr"]
            ue.ewma_dr        = ue_state["ewma_dr"]

        # restore BSs
        for bs_state in state["base_stations"]:
            bs = next(b for b in self.base_stations if b.id == bs_state["id"])
            bs.allocated_resources = bs_state["allocated_resources"].copy()
            # restore PRB maps too
            bs.rb_allocation = {int(uid): prbs.copy() 
                                for uid, prbs in bs_state["rb_allocation"].items()}
            bs.load        = bs_state["load"]
            bs.capacity    = bs_state["capacity"]
            bs.reuse_color = bs_state.get("reuse_color", bs.reuse_color)
            bs.frequency   = bs_state.get("frequency", bs.frequency)

        # # restore histories & counters
        # self.load_history      = {int(bs_id): hist.copy() 
        #                         for bs_id, hist in state.get("load_history", {}).items()}
        # self.handover_counts   = state.get("handover_counts", {}).copy()
        # self.prev_associations = state.get("prev_associations", {}).copy()
        # self.step_count        = state.get("step_count", self.step_count)
        # self.current_step      = state.get("current_step", self.current_step)    
    


    
    # def _update_system_metrics(self):
    #     # 1) Recompute loads from PF‐shared allocations
    #     for bs in self.base_stations:
    #         bs.calculate_load()

    #     # 2) Recompute each UE’s SINR using the multi‐cell helper
    #     for ue in self.ues:
    #         if ue.associated_bs is not None:
    #             sinr_vec = self._calculate_sinrs(ue)     # NumPy vector of SINRs
    #             ue.sinr = sinr_vec[ue.associated_bs]
    #         else:
    #             ue.sinr = -np.inf
    def _update_system_metrics(self):
        """
        Refresh per-TTI system metrics:
        1) Updates each BS.load via calculate_load() (throughput or RB usage).
        2) Updates each UE.sinr as the average per-RB SINR over its allocated RBs.
        3) (Optional) Append histories for PRBS masks and SINR, if needed.
        """
        # print("Updating System Metrics....")
        # 1) Recompute loads
        for bs in self.base_stations:
            bs.calculate_load()
            # print(f"For Update System Metrics to {bs.id}, Load :{bs.load}")
        # 2) Update UE SINRs based on actual RB allocations
        ue_iter = self.ues.values() if isinstance(self.ues, dict) else self.ues
        for ue in ue_iter:
            if ue.associated_bs is not None:
                bs = next(b for b in self.base_stations if b.id == ue.associated_bs)

                # Get per-RB SINRs
                sinr_rb = bs.snr_per_rb(ue)

                # Determine RBs allocated this TTI
                rb_list = bs.rb_allocation.get(ue.id, [])
                # print (f"RB List is:{rb_list}")
                if rb_list:
                    # Mean SINR over allocated RBs
                    ue.sinr = float(np.mean(sinr_rb[rb_list]))
                else:
                    ue.sinr = -np.inf
            else:
                ue.sinr = -np.inf
    
    def apply_solution(self, solution):
        """
        Apply a user↔BS association mapping and run the PRB-by-PRB PF scheduler
        on each BS for its set of UEs.
        """
        # --- 1) Normalize solution dict ---
        # print("Applying Solution to Environment......")
        if isinstance(solution, np.ndarray):
            sol_dict = {bs.id: [] for bs in self.base_stations}
            for ue_idx, bs_id in enumerate(solution.astype(int)):
                sol_dict[int(bs_id)].append(ue_idx)
            solution = sol_dict

        # --- 2) Validate BS and UE indices (as before) ---
        valid_bs_ids = {int(bs.id) for bs in self.base_stations}
        num_ues = len(self.ues)
        for bs_id, ue_list in solution.items():
            if int(bs_id) not in valid_bs_ids:
                raise ValueError(f"Invalid BS ID {bs_id}")
            for ue_id in ue_list:
                if ue_id < 0 or ue_id >= num_ues:
                    raise IndexError(f"UE index {ue_id} out of range")

        # --- 3) Clear out old allocs & associations ---
        for bs in self.base_stations:
            bs.rb_allocation = {ue_id: [] for ue_id in range(len(self.ues))}
            bs.allocated_resources.clear()
        ue_iter = self.ues.values() if isinstance(self.ues, dict) else self.ues
        for ue in ue_iter:
            ue.associated_bs = None

        # --- 4) Apply new associations ---
        for bs_id, ue_list in solution.items():
            bs = next(b for b in self.base_stations if b.id == int(bs_id))
            # Mark UEs as “in cell”
            for ue_id in ue_list:
                self.ues[ue_id].associated_bs = bs.id

        # --- 5) Run PF scheduler on each BS ---
        for bs in self.base_stations:
            # Only schedule the UEs currently associated
            ue_iter = self.ues.values() if isinstance(self.ues, dict) else self.ues    
            active_ues = {ue.id: ue for ue in ue_iter if ue.associated_bs == bs.id}
            bs.ues = active_ues
            bs.allocate_prbs()    # your PRB-by-PRB PF method
            
        # After apply_solution(solution) and allocate_prbs() have run:

        # Build a map ue_id → { bs_id, prbs: [...], sinrs_per_prb: [...] }
        alloc_details = {}

        # We need a fast lookup of UE objects by id
        ue_map = {ue.id: ue for ue in (self.ues.values() if isinstance(self.ues, dict) else self.ues)}

        for bs in self.base_stations:
            # bs.rb_allocation maps ue_id → list of PRB indices
            for ue_id, prb_list in bs.rb_allocation.items():
                if not prb_list:
                    continue

                # Compute the full SINR array for this UE
                ue = ue_map[ue_id]
                sinr_array = bs.snr_per_rb(ue)  # length = self.num_rbs

                # Extract only the SINRs on the allocated PRBs
                sinrs_on_prbs = sinr_array[prb_list]

                alloc_details[ue_id] = {
                    "bs_id": bs.id,
                    "prbs": prb_list,
                    "sinrs": sinrs_on_prbs.tolist()
                }

        # # Now you can inspect alloc_details, for example:
        # for ue_id, info in alloc_details.items():
        #     print(f"UE {ue_id} on BS {info['bs_id']},PRBs: {info['prbs']},SINRs (linear): {info['sinrs']}")
            

        # --- 6) Track load history & handovers ---
        for bs in self.base_stations:
            self.load_history[bs.id].append(bs.load)
        ue_iter = self.ues.values() if isinstance(self.ues, dict) else self.ues
        for ue in ue_iter:
        # for ue in self.ues:
            old = self.prev_associations[ue.id]
            new = ue.associated_bs
            if old is not None and new is not None and old != new:
                self.handover_counts[ue.id] += 1
            self.prev_associations[ue.id] = new
        self.step_count += 1

        # --- 7) Refresh per-UE SINR & BS loads ---
        self._update_system_metrics()

    
    def evaluate_detailed_solution(self, solution, alpha=0.1, beta=0.1):
        """
        Apply a candidate user-association solution, compute detailed performance metrics,
        then restore state. Returns metrics including average throughput in GB/s.
        """
        # # Update UE positions using MRWP mobility model
        # for ue in self.ues:
        #     ue.update_position()
        
        # 1) Snapshot current state
        original = self.get_state_snapshot()

        # 2) Apply the proposed associations
        self.apply_solution(solution)

        # 3) Ensure loads and SINRs are up-to-date
        for bs in self.base_stations:
            bs.calculate_load()
        ue_iter = self.ues.values() if isinstance(self.ues, dict) else self.ues
        for ue in ue_iter:
            if ue.associated_bs is not None:
                # debug SINR for the first 3 UEs
                sinr_vec = self._calculate_sinrs(ue) #, debug=(ue.id < 3))
                ue.sinr = sinr_vec[ue.associated_bs]
            else:
                ue.sinr = -np.inf

        # 4) Throughputs per UE from actual shared allocations (bits/sec)
        throughputs = np.zeros(self.num_ue, dtype=np.float32)
        for bs in self.base_stations:
            for ue_id, dr in bs.allocated_resources.items():
                throughputs[ue_id] = dr

        # Convert to GB/s for display and metrics
        throughputs_Mbps = throughputs / 1e6  # bits/sec → bytes/sec → Gb/sec
        throughputs_Gbps = throughputs / 1e9
        avg_throughput_Mbps = throughputs_Mbps.mean()
        sum_throughputs=throughputs_Gbps.sum()
        # # Debug: print a few sample UE stats in GB/s
        # for ue_id in throughputs.argsort()[-5:]:
        #     ue = self.ues[ue_id]            
        #     lin_snr = ue.sinr
        #     snr_db  = 10*np.log10(lin_snr + 1e-12)
        #     r_gbps = throughputs_Gbps[ue_id]
        #     print(f" UE {ue_id}: assoc→BS{ue.associated_bs}, "
        #         f"SINR={snr_db:.2f} dB, Rate={r_gbps:.3f} Gb/s")        
        ue_iter = self.ues.values() if isinstance(self.ues, dict) else self.ues
        # 5) Compute other metrics
        fitness     = self.calculate_reward()          # global reward
        avg_sinr    = np.mean([ue.sinr for ue in ue_iter])
        avg_sinr_db = 10 * np.log10(avg_sinr)
        fairness    = (throughputs.sum()**2) / (len(throughputs) * np.sum(throughputs**2) + 1e-9)
        load_var    = np.var([bs.load for bs in self.base_stations])
        bs_loads    = [bs.load for bs in self.base_stations]
        
        # 6) Handover rate
        total_handover_events = sum(self.handover_counts.values())
        ho_rate_per_step = total_handover_events / (self.num_ue * self.step_count)
        
        # 7) Load‐distribution quantiles (Gbps)
        # concatenate all BS‐load histories (bps), convert to Gbps
        all_loads_bps = np.hstack([self.load_history[bs.id] for bs in self.base_stations])        
        all_loads_gbps = all_loads_bps / 1e9
        q10, q50, q90 = np.quantile(all_loads_gbps, [0.1, 0.5, 0.9])

        # 6) Restore original state
        self.set_state_snapshot(original)
        
        # 7) Return detailed report (throughput in GB/s)
        return {
            "fitness": float(fitness),
            "average_sinr": float(avg_sinr_db),
            "average_throughput": float(avg_throughput_Mbps),
            "sum_throughput":float(sum_throughputs),
            "fairness": float(fairness),
            "load_variance": float(load_var),
            "bs_loads": bs_loads,
            "handover_rate": ho_rate_per_step,
            "load_quantiles_Gbps": {"10th": q10, "50th": q50, "90th": q90}

        }

    

        
# env = NetworkEnvironment({"num_ue": 3, "num_bs": 2})
# obs, _ = env.reset()
# print(obs["ue_0"].shape)  # Should be (2*2 + 1)=5

# actions = {"ue_0": 1, "ue_1": 0, "ue_2": 1}  # Each UE selects a BS index
# next_obs, rewards, dones, _ = env.step(actions)
# print(next_obs, rewards, dones, _ )
# env1 = NetworkEnvironment({"num_ue": 10, "num_bs": 3})
# positions1 = [ue.position for ue in env1.ues]

# env2 = NetworkEnvironment({"num_ue": 10, "num_bs": 3})
# positions2 = [ue.position for ue in env2.ues]

# assert np.allclose(positions1, positions2)  # ✅ Should pass now
# print(np.allclose(positions1, positions2))  # Should print: True
