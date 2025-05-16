        
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
        # Scheduling fields:
        self.associated_bs = None
        self.sinr          = -np.inf
        self.ewma_dr       = 1e6
        self.ewma_alpha    = ewma_alpha
        self.rx_gain = float(rx_gain_dbi)  # dBi
    def _draw_waypoint(self):
        # uniformly anywhere in the 100×100 area
        return np.random.uniform(0, 100, size=2).astype(np.float32)

    def update_position(self, delta_time=1.0):
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
    def __init__(self, id, position, frequency, bandwidth, height=50.0, reuse_color=None,tx_power_dbm=30.0,tx_gain_dbi=8.0,bf_gain_dbi=20.0):
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
        # Resource block parameters
        self.rb_bandwidth = 180e3  # Hz (typical 5G/6G subcarrier spacing × 12)
        self.num_rbs = int(self.bandwidth / self.rb_bandwidth)  # Number of available RBs
        
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
    
    def path_loss(self, distance, d0=1.0, n=2.0, sigma=0.0):
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
        fspl_d0 = 20 * np.log10(4 * np.pi * d0 * self.frequency / c)

        # 2) Distance‐dependent term
        pl_mean = fspl_d0 + 10 * n * np.log10(distance / d0)

        # 3) Add shadow fading if desired
        shadow = np.random.randn() * sigma if sigma > 0 else 0.0

        return pl_mean + shadow

    def noise_mW(self):
        # Thermal noise: -174 dBm/Hz + 10*log10(BW_Hz)
        noise_dbm = -174 + 10 * np.log10(self.bandwidth )
        return 10 ** (noise_dbm / 10)
    
    def _calculate_local_interference(self, neighbor_dist=100.0):
        """Calculate interference at this BS from all other BSs within neighbor_dist."""
        interference = 0.0
        for other in self.base_stations:
            if other.id == self.id:
                continue
            d = np.linalg.norm(self.position - other.position)
            if d < neighbor_dist:
                # received power from interfering BS
                prx_int = other.received_power_mW(self.position)
                interference += prx_int
        return interference           
    
    
    def received_power_mW(self, ue_pos, ue_rx_gain=None):# , ue_bf_gain_dbi=0.0
        d = np.linalg.norm(self.position - ue_pos)
        # The code is calculating the path loss using a function `path_loss` with the distance `d` as
        # an input parameter and storing the result in the variable `L`.
        L  = self.path_loss(d)
        G_tx = self.tx_gain + self.bf_gain      # total transmit gain
        G_rx = ue_rx_gain if ue_rx_gain is not None else 0.0 # (ue_rx_gain or 0.0) + ue_bf_gain_dbi  # UE may also beam-form
        p_rx_dbm = self.tx_power + G_tx + G_rx - L
        return 10**(p_rx_dbm/10)
    
    def snr_linear(self, ue):
        """
        Compute the linear SINR for this BS → ue link, including
        Tx/Rx gains, co-channel interference, and thermal noise.
        """
        # 1) desired signal power (mW)
        signal_mW = self.received_power_mW(ue.position, ue.rx_gain)#, ue.bf_gain

        # 2) co‐channel interference: sum mW from other BSs on same reuse_color
        interference_mW = 0.0
        for other in self.base_stations:
            if other.id == self.id:
                continue
            if other.reuse_color != self.reuse_color:
                continue
            interference_mW += other.received_power_mW(ue.position, ue.rx_gain)

        # 3) noise power (mW)
        noise_mW = self.noise_mW()

        # 4) linear SINR
        return signal_mW / (interference_mW + noise_mW + 1e-12)
    
    def allocate_prbs(self):
        """
        Optimized version of the PRB allocation method that maintains fairness guarantees
        while reducing computational complexity and redundant operations.
        """
        # Handle empty case
        if not self.ues:
            self.rb_allocation = {}
            self.allocated_resources = {}
            self.calculate_load()
            return
            
        # Initialize allocation and delivered bits
        self.rb_allocation = {ue_id: [] for ue_id in self.ues}
        delivered_bits = {ue_id: 0.0 for ue_id in self.ues}
        
        # Fast path for single UE case
        if len(self.ues) == 1:
            ue_id = next(iter(self.ues))
            self.rb_allocation[ue_id] = list(range(self.num_rbs))         
            # Precompute rate for this single UE
            sinr_array = self.snr_per_rb(self.ues[ue_id])
            rate_array = self.rb_bandwidth * np.log2(1 + sinr_array)
            rate_map = {ue_id: rate_array}  # Add this line
            delivered_bits[ue_id] = np.sum(rate_array)
        else:
            # --------------------------------------------------
            # OPTIMIZATION 1: Efficient Precomputation with Caching
            # --------------------------------------------------
            
            # Precompute SINR arrays, rates, and best PRB ordering for each UE - all at once
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
                
                # Precompute sorted PRB indices (best to worst) - used multiple times
                sorted_prb_indices[ue_id] = np.argsort(-sinr_array)
            
            # --------------------------------------------------
            # OPTIMIZATION 2: Efficient Clustering
            # --------------------------------------------------
            
            # Sort UEs by average SINR (once, not repeatedly)
            sorted_ues = sorted(avg_sinr_db.items(), key=lambda x: x[1])
            
            # Use a more efficient approach to create SINR clusters
            sinr_clusters = []
            current_cluster = [sorted_ues[0][0]]
            current_base_sinr = sorted_ues[0][1]
            
            for ue_id, sinr in sorted_ues[1:]:
                if sinr - current_base_sinr <= 3.0:  # Within 3dB of cluster base
                    current_cluster.append(ue_id)
                else:
                    # Start a new cluster
                    sinr_clusters.append(current_cluster)
                    current_cluster = [ue_id]
                    current_base_sinr = sinr
            
            # Add the last cluster
            if current_cluster:
                sinr_clusters.append(current_cluster)
            
            # --------------------------------------------------
            # OPTIMIZATION 3: Efficient PRB Tracking with NumPy
            # --------------------------------------------------
            
            # Use boolean array for PRB availability instead of sets (much faster)
            available_prbs = np.ones(self.num_rbs, dtype=bool)  # All PRBs available initially
            
            # --------------------------------------------------
            # OPTIMIZATION 4: Efficient Cluster PRB Allocation
            # --------------------------------------------------
            
            # Allocate PRBs to clusters (similar approach, optimized)
            total_ues = len(self.ues)
            cluster_prb_allocation = {}
            remaining_prbs_count = self.num_rbs
            
            # Base allocation: proportional to cluster size with bias toward lower SINR clusters
            for i, cluster in enumerate(sinr_clusters):
                cluster_size = len(cluster)
                
                # Base allocation proportional to cluster size
                base_allocation = int(self.num_rbs * (cluster_size / total_ues))
                
                # Add bias toward lower SINR clusters (more PRBs for worse conditions)
                bias_factor = 1.0 + max(0, 0.1 * (len(sinr_clusters) - i - 1))
                
                # Final allocation with bias (ensure we don't exceed remaining PRBs)
                cluster_allocation = min(
                    int(base_allocation * bias_factor),
                    remaining_prbs_count
                )
                
                # Store allocation for this cluster
                cluster_prb_allocation[i] = cluster_allocation
                remaining_prbs_count -= cluster_allocation
            
            # Distribute any remaining PRBs to clusters starting from lowest SINR
            if remaining_prbs_count > 0:
                for i in range(len(sinr_clusters)):
                    if remaining_prbs_count <= 0:
                        break
                    cluster_prb_allocation[i] += 1
                    remaining_prbs_count -= 1
            
            # --------------------------------------------------
            # OPTIMIZATION 5: Efficient Intra-Cluster Allocation
            # --------------------------------------------------
            
            # Process each cluster
            for cluster_idx, cluster in enumerate(sinr_clusters):
                cluster_prbs_needed = cluster_prb_allocation[cluster_idx]
                
                # Skip empty clusters
                if cluster_prbs_needed <= 0:
                    continue
                
                # First pass: minimum allocation per UE in this cluster
                min_prbs_per_ue = max(1, cluster_prbs_needed // len(cluster))
                
                # Allocate minimum PRBs to each UE in this cluster (in batch)
                for ue_id in cluster:
                    # Get pre-sorted PRB indices for this UE
                    prb_preference = sorted_prb_indices[ue_id]
                    
                    # Find best available PRBs for this UE
                    allocated_count = 0
                    for prb_idx in prb_preference:
                        if available_prbs[prb_idx] and allocated_count < min_prbs_per_ue:
                            # Allocate this PRB
                            self.rb_allocation[ue_id].append(prb_idx)
                            available_prbs[prb_idx] = False  # Mark as used
                            delivered_bits[ue_id] += rate_map[ue_id][prb_idx]
                            allocated_count += 1
                            cluster_prbs_needed -= 1
                            
                            # Early exit if we've allocated enough
                            if allocated_count >= min_prbs_per_ue:
                                break
                
                # Second pass: distribute remaining PRBs within cluster using modified PF
                if cluster_prbs_needed > 0 and np.any(available_prbs):
                    # Calculate intra-cluster fairness metrics (once, not repeatedly)
                    
                    # Get current rates within cluster
                    cluster_rates = {}
                    for ue_id in cluster:
                        if self.rb_allocation[ue_id]:
                            allocated_prbs = self.rb_allocation[ue_id]
                            cluster_rates[ue_id] = sum(rate_map[ue_id][prb] for prb in allocated_prbs)
                        else:
                            cluster_rates[ue_id] = 0.0
                    
                    # Find max rate within cluster for normalization
                    max_cluster_rate = max(cluster_rates.values()) if cluster_rates else 1.0
                    max_cluster_rate = max(max_cluster_rate, 1.0)  # Ensure non-zero
                    
                    # Calculate fairness factor within cluster (once, not in loop)
                    fairness_factor = {}
                    for ue_id in cluster:
                        normalized_rate = cluster_rates[ue_id] / max_cluster_rate
                        # Strong inverse relationship - lower rates get much higher priority
                        fairness_factor[ue_id] = 1.0 / (0.2 + 0.8 * normalized_rate)
                    
                    # Get remaining PRBs indices as a NumPy array (faster than list conversion)
                    remaining_prb_indices = np.where(available_prbs)[0]
                    
                    # Allocate remaining PRBs efficiently
                    for prb in remaining_prb_indices[:cluster_prbs_needed]:
                        # Find best UE for this PRB in the cluster
                        best_metric = -float('inf')
                        best_ue = None
                        
                        for ue_id in cluster:
                            # Avoid division by zero with a safe minimum value
                            ewma = max(self.ues[ue_id].ewma_dr, 1e-6)
                            
                            # Calculate metric with strong fairness bias
                            metric = rate_map[ue_id][prb] / ewma * fairness_factor[ue_id]
                            
                            if metric > best_metric:
                                best_metric = metric
                                best_ue = ue_id
                        
                        # Allocate PRB to best UE
                        if best_ue is not None:
                            self.rb_allocation[best_ue].append(prb)
                            available_prbs[prb] = False  # Mark as used
                            delivered_bits[best_ue] += rate_map[best_ue][prb]
            
            # --------------------------------------------------
            # OPTIMIZATION 6: Efficient Global PF for Remaining PRBs
            # --------------------------------------------------
            
            # Allocate any remaining PRBs using standard PF with fairness
            remaining_prb_indices = np.where(available_prbs)[0]
            if len(remaining_prb_indices) > 0:
                # Calculate current rates for fairness (once, not in loop)
                current_rates = {}
                for ue_id in self.ues:
                    if self.rb_allocation[ue_id]:
                        allocated_prbs = self.rb_allocation[ue_id]
                        current_rates[ue_id] = sum(rate_map[ue_id][prb] for prb in allocated_prbs)
                    else:
                        current_rates[ue_id] = 0.0
                
                # Calculate max rate for normalization
                max_rate = max(current_rates.values()) if current_rates else 1.0
                max_rate = max(max_rate, 1.0)  # Ensure non-zero
                
                # Calculate inverse rate fairness factor (once, not in loop)
                fairness_factor = {}
                for ue_id in self.ues:
                    normalized_rate = current_rates[ue_id] / max_rate
                    # Lower rates get higher priority (inverse relationship)
                    fairness_factor[ue_id] = 1.0 / max(0.1, normalized_rate)
                
                # Allocate remaining PRBs
                for prb in remaining_prb_indices:
                    # Find best UE for this PRB across all UEs
                    best_metric = -float('inf')
                    best_ue = None
                    
                    for ue_id in self.ues:
                        # Avoid division by zero with a safe minimum value
                        ewma = max(self.ues[ue_id].ewma_dr, 1e-6)
                        
                        # Calculate metric with fairness factor
                        metric = rate_map[ue_id][prb] / ewma * fairness_factor[ue_id]
                        
                        if metric > best_metric:
                            best_metric = metric
                            best_ue = ue_id
                    
                    # Allocate PRB to best UE
                    if best_ue is not None:
                        self.rb_allocation[best_ue].append(prb)
                        available_prbs[prb] = False  # Mark as used
                        delivered_bits[best_ue] += rate_map[best_ue][prb]
            
            # --------------------------------------------------
            # OPTIMIZATION 7: Efficient Zero-PRB UE Check
            # --------------------------------------------------
            
            # Final check for UEs with zero PRBs (much less likely with optimized approach)
            zero_prb_ues = [ue_id for ue_id, prbs in self.rb_allocation.items() if not prbs]
            if zero_prb_ues:
                # Identify UEs with many PRBs (do this once, not repeatedly)
                ue_prb_counts = [(ue_id, len(prbs)) for ue_id, prbs in self.rb_allocation.items() if prbs]
                ue_prb_counts.sort(key=lambda x: x[1], reverse=True)
                
                # Redistribute one PRB from each UE with many PRBs
                for zero_ue in zero_prb_ues:
                    if ue_prb_counts and ue_prb_counts[0][1] > 1:
                        donor_ue, _ = ue_prb_counts[0]
                        
                        # Take one PRB from donor
                        prb = self.rb_allocation[donor_ue].pop()
                        
                        # Give to zero PRB UE
                        self.rb_allocation[zero_ue].append(prb)
                        delivered_bits[zero_ue] += rate_map[zero_ue][prb]
                        delivered_bits[donor_ue] -= rate_map[donor_ue][prb]
                        
                        # Update PRB count list
                        ue_prb_counts[0] = (donor_ue, ue_prb_counts[0][1] - 1)
                        ue_prb_counts.sort(key=lambda x: x[1], reverse=True)
        
        # --------------------------------------------------
        # OPTIMIZATION 8: Efficient EWMA Update
        # --------------------------------------------------
        
        # Update EWMA throughput for each UE - ensure minimum value
        for ue_id, ue in self.ues.items():
            # Ensure EWMA never gets too small
            ue.update_ewma(delivered_bits[ue_id])
            if ue.ewma_dr < 1e-6:  # Set a minimal EWMA to prevent division issues
                ue.ewma_dr = 1e-6
        
        # Calculate allocated resources in bps
        self.allocated_resources = {}
        for ue_id, prbs in self.rb_allocation.items():
            if prbs:  # Only calculate for UEs with allocated PRBs
                # Use more efficient calculation of allocated resources
                self.allocated_resources[ue_id] = sum(rate_map[ue_id][prb] for prb in prbs)
            else:
                self.allocated_resources[ue_id] = 0.0
        
        # Calculate system load
        self.calculate_load()

    
    
    
    def calculate_capacity_rb_based(self, sample_points=100, overhead_factor=0.8):
        """
        Estimate BS capacity based on sample-driven average spectral efficiency.
        - sample_points: number of random spatial samples in cell area
        - overhead_factor: fraction of resources available for user data
        Returns capacity in Mbps.
        """
        total_se = 0.0  # sum of spectral efficiencies (bits/s/Hz)
        # Determine cell radius from BS positions (max distance)
        cell_radius = max(np.linalg.norm(bs.position - self.position)
                          for bs in self.base_stations)

        for _ in range(sample_points):
            # Uniformly sample a point in the circular cell
            u = np.random.uniform(0, 1)
            r = np.sqrt(u) * cell_radius
            theta = np.random.uniform(0, 2 * np.pi)
            sample_point = self.position + np.array([r * np.cos(theta),
                                                   r * np.sin(theta)], dtype=np.float32)

            # Desired signal (mW)
            d = np.linalg.norm(sample_point - self.position)
            pl = self.path_loss(d)
            prx = 10 ** ((self.tx_power + self.tx_gain + self.bf_gain - pl) / 10)

            # Co-channel interference (mW)
            interf = 0.0
            for other in self.base_stations:
                if other.id != self.id and other.reuse_color == self.reuse_color:
                    d_int = np.linalg.norm(sample_point - other.position)
                    pl_int = other.path_loss(d_int)
                    interf += 10 ** ((other.tx_power + other.tx_gain + other.bf_gain - pl_int) / 10)

            # Noise (mW)
            noise = self.noise_mW()
            sinr = prx / (interf + noise + 1e-12)
            total_se += np.log2(1 + sinr)

        # Compute average spectral efficiency (bits/s/Hz)
        avg_se = total_se / sample_points

        # Per-RB capacity (bits/s)
        rb_capacity = self.rb_bandwidth * avg_se

        # Total capacity (bits/s) with overhead
        total_bps = rb_capacity * self.num_rbs * overhead_factor

        # Convert to Mbps
        self.capacity = total_bps / 1e6
        return self.capacity
    def snr_per_rb(self, ue):
        """
        Compute per-RB linear SINR array for a given UE.
        Returns a numpy array of length self.num_rbs.
        """
        sinr_rb = np.empty(self.num_rbs, dtype=np.float32)
        # Pre-compute noise per RB (assuming noise_mW returns total per RB noise)
        noise_rb = self.noise_mW()

        # For flat fading: the received power is same on every RB, so compute once
        prx = self.received_power_mW(ue.position, ue.rx_gain)

        # Compute interference per RB: sum of mW from co-channel BSs
        interf = 0.0
        for other in self.base_stations:
            if other.id == self.id or other.reuse_color != self.reuse_color:
                continue
            interf += other.received_power_mW(ue.position, ue.rx_gain)

        # Fill array
        sinr_val = prx / (interf + noise_rb + 1e-12)
        sinr_rb.fill(sinr_val)
        return sinr_rb
    
    def data_rate_unshared(self, ue):
        """
        Returns the instantaneous Shannon capacity of UE if alone on the BS.
        - self.bandwidth in Hz
        - snr_linear returns a linear ratio (not in dB)
        -> result is in bits/sec
        """
        snr = self.snr_linear(ue)
        return self.bandwidth * np.log2(1 + snr)

    def priority(self, ue, alpha=1.0, beta=1.0, eps=1e-6):
        T_inst = self.data_rate_unshared(ue) #bits/sec (or Mbps, if you divided)
        R_hist = ue.ewma_dr   # same units as T_inst
        return (T_inst ** alpha) / (R_hist ** beta + eps)
    
    def data_rate_shared(self, ue, alpha=1.0, beta=1.0, gamma=0.5):
        """Enhanced PF scheduler incorporating demand"""
        # 1) Compute priority with demand factor
        weights = []
        ue_demands = {}
        
        for other_ue_id in self.allocated_resources:
            other_ue = self.ues[other_ue_id]
            # Base priority using PF
            w_pf = self.priority(other_ue, alpha, beta)
            
            # Demand factor: prioritize UEs whose demand is not yet met
            demand_factor = min(1.0, other_ue.demand / (other_ue.ewma_dr + 1e-6))**gamma
            
            # Final priority
            w = w_pf * demand_factor
            weights.append(w)
            ue_demands[other_ue_id] = other_ue.demand
        
        W = sum(weights) + 1e-9
        
        # 2) Fraction for THIS UE with demand consideration
        w_i_pf = self.priority(ue, alpha, beta)
        demand_factor_i = min(1.0, ue.demand / (ue.ewma_dr + 1e-6))**gamma
        w_i = w_i_pf * demand_factor_i
        
        fraction = w_i / W
        
        # 3) Shared rate = fraction × instantaneous capacity
        T_inst = self.data_rate_unshared(ue)
        
        # 4) Cap allocated rate at demand if desired
        allocated_rate = fraction * T_inst
        # Uncomment to cap at demand:
        # allocated_rate = min(allocated_rate, ue.demand)
        
        return allocated_rate
    
class NetworkEnvironment(MultiAgentEnv):
    @staticmethod
    def generate_hex_positions(num_bs, width=100.0, height=100.0):
        """
        Generate base station positions in a hexagonal pattern that automatically
        scales spacing based on the number of stations, ensuring even distribution
        across the entire map area.
        """
        if num_bs < 1:
            return []

        # Calculate spacing based on area per BS
        area_per_bs = (width * height) / num_bs
        pitch = math.sqrt((2 * area_per_bs) / math.sqrt(3))
        
        # Prevent excessive spacing for very small numbers
        max_pitch = min(width, height) / 2
        pitch = min(pitch, max_pitch)
        
        # Ensure minimum spacing for readability
        pitch = max(pitch, 5.0)

        # Calculate grid dimensions to cover entire map
        hex_height = pitch * math.sin(math.radians(60))
        n_cols = int(math.ceil(width / pitch)) + 2
        n_rows = int(math.ceil(height / hex_height)) + 2

        # Center the grid
        total_width = (n_cols-1) * pitch
        total_height = (n_rows-1) * hex_height
        x_offset = (width - total_width) / 2
        y_offset = (height - total_height) / 2

        # Generate all potential positions
        positions = []
        for row in range(n_rows):
            y = y_offset + row * hex_height
            x_start = x_offset + (pitch/2 if row % 2 else 0)
            
            for col in range(n_cols):
                x = x_start + col * pitch
                if 0 <= x <= width and 0 <= y <= height:
                    positions.append((x, y))

        # Sort by distance from center with spiral pattern
        center = (width/2, height/2)
        positions.sort(
            key=lambda p: (
                math.hypot(p[0]-center[0], p[1]-center[1]),  # Primary: distance
                -math.atan2(p[1]-center[1], p[0]-center[0])   # Secondary: angle
            )
        )

        return [((x, y)) for x, y in positions[:num_bs]]

    
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
        
        self.step_count       = 0        
        self.base_stations = []
        # Calculating hexagonal positions for BS
        # 1) define your reuse pattern and carrier frequencies
        colors = ["A", "B", "C"]
        carrier_freqs = {
            "A": 140e9,   # GHz, e.g. 140 GHz
            "B": 141e9,   # GHz
            "C": 142e9    # GHz
        }
        bandwidth = 10e9  # Hz, e.g. 10 GHz per carrier slice

        # 2) get (row, col, pos) tuples
        grid = self.generate_hex_positions(self.num_bs, width=100.0, height=100.0)

        # 3) instantiate BS with reuse_color & per‐slice frequency
        self.base_stations = []
        # Then update your BS instantiation loop:
        for i, pos in enumerate(grid):
            color = colors[i % len(colors)]  # Simple rotating color scheme
            # Or for spatial color distribution:
            # color = colors[int((pos[0]/100 + pos[1]/100) * len(colors)) % len(colors)]
            freq = carrier_freqs[color]
            bs = BaseStation(
                id=i,
                position=pos,
                frequency=freq,
                bandwidth=bandwidth,
                reuse_color=color
                # ... other parameters
            )
        # for i, (row, col, pos) in enumerate(grid):
        #     color = colors[(row + 2*col) % len(colors)]
        #     freq  = carrier_freqs[color]
        #     bs = BaseStation(
        #         id=i,
        #         position=pos,
        #         frequency=freq,
        #         bandwidth=bandwidth,
        #         reuse_color=color                
        #     )
            self.base_stations.append(bs)
        print(f"[NetworkEnvironment] num_bs = {config.get('num_bs')}")     
        self.ues = [
            UE(
                id=i,
                position=np.random.uniform(0, 100, size=2).astype(np.float32),
                demand=np.random.randint(50, 200),
                v_min=0.5,
                v_max=1.5,
                pause_min=1.0,
                pause_max=5.0,
                ewma_alpha=0.9
            )
            for i in range(self.num_ue)
        ]
        print(f"Created {len(self.ues)} UEs")
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
        obs_dim = 3*self.num_bs + 1 + (self.num_bs + 1) # SINRs + BS loads + BS Utilizations + own demand + Connected
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

        
    # def observation_space(self, agent_id=None):
    #     if agent_id is not None:
    #         return self.observation_space[agent_id]
    #     return self.observation_space

    # def action_space(self, agent_id=None):
    #     if agent_id is not None:
    #         return self.action_space[agent_id]
    #     return self.action_space

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

    
    
    # def pick_best_bs(self, ue, w1=0.7, w2=0.3):
    #     unshared = np.array([bs.data_rate_unshared(ue) for bs in self.base_stations])
    #     max_unsh = unshared.max() + 1e-9
    #     loads = np.array([bs.load for bs in self.base_stations])
    #     caps  = np.array([bs.capacity for bs in self.base_stations])
    #     load_factor = 1 - loads / caps
    #     scores = w1 * (unshared / max_unsh) + w2 * load_factor
    #     return int(np.argmax(scores))

    # def reset(self, seed=None, options=None):
    #     self.current_step = 0
    #     for ue in self.ues:
    #         ue.position = np.random.uniform(0,100,2).astype(np.float32)
    #         ue.associated_bs = None
    #         ue.sinr = -np.inf
    #         ue.ewma_dr = 0.0
    #     return self._get_obs(), {}    
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
            print(f"Connected Users : {connected_count} Users")
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
            # now total_throughput is in bits/s per Hz—if you want Gbps, multiply by your PRB_bw and num_prbs:
            total_throughput_gbps = total_throughput * self.rb_bandwidth * len(self.base_stations[0].rb_allocation) / 1e9

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
            info       = self._build_info_dict()
            self.current_step += 1

            return obs, rewards, terminated, truncated, info
        except Exception as e:        
            print(f"ERROR in step: {e}")
            import traceback
            print(traceback.format_exc())
            # Return a safe default response
            return self._get_obs(), {f"ue_{ue.id}": 0.0 for ue in self.ues}, {"__all__": False}, {"__all__": True}, {"__common__": {"error": str(e)}} 
    # def step(self, actions: Dict[str, int]):
    #     # Add timing for performance analysis
    #     print(f"Step called with {len(actions)} actions")
    #     try:
    #         start_time = time.time()            
    #         connected_count = 0
    #         handover_count = 0
    #         for i, ue in enumerate(self.ues):
    #             print(f"Before step - UE {i}: associated_bs={ue.associated_bs}")
    #         for i, bs in enumerate(self.base_stations):
    #             print(f"Before step - BS {i}: load={bs.load}, allocated_resources={bs.allocated_resources}")
    #         # 1) Process each UE’s action
    #         for agent_id, a in actions.items():
    #             i = int(agent_id.split("_")[1])
    #             ue = self.ues[i]

    #             # decode new BS index
    #             new_bs = None if a == 0 else (a - 1)
    #             old_bs = ue.associated_bs

    #             if new_bs != old_bs:
    #                 # remove from old BS
    #                 if old_bs is not None:
    #                     self.base_stations[old_bs].allocated_resources.pop(ue.id, None)
    #                     self.base_stations[old_bs].calculate_load()

    #                 # try admit to new BS
    #                 if new_bs is not None:
    #                     bs = self.base_stations[new_bs]
    #                     dr = bs.data_rate_shared(ue)
    #                     if bs.load + dr <= bs.capacity:
    #                         bs.allocated_resources[ue.id] = dr
    #                         bs.calculate_load()
    #                         ue.associated_bs = new_bs
    #                         connected_count += 1
    #                     else:
    #                         # capacity full → revert to old
    #                         ue.associated_bs = old_bs
    #                         if old_bs is not None:
    #                             connected_count += 1
    #                 handover_count += 1
    #             else:
    #                 # stayed put
    #                 if ue.associated_bs is not None:
    #                     connected_count += 1

            
    #         # 3) Update SINR & EWMA
    #         for ue in self.ues:
    #             # full multi-cell SINR vector
    #             sinr_vec = self._calculate_sinrs(ue)
    #             if ue.associated_bs is not None:
    #                 ue.sinr = sinr_vec[ue.associated_bs]
    #                 # update EWMA with allocated rate
    #                 self.base_stations[ue.associated_bs].calculate_load()
    #                 measured = self.base_stations[ue.associated_bs].allocated_resources.get(ue.id, 0.0)
    #                 ue.update_ewma(measured)
    #             else:
    #                 ue.sinr = -np.inf
            
    #         # Calculate rewards efficiently
    #         rewards = {}
    #         total_reward = 0
    #         for bs in self.base_stations:
    #             print(f"BS {bs.id}: load={bs.load}, capacity={bs.capacity}, ratio={bs.load/bs.capacity if bs.capacity>0 else 'inf'}")
            
    #         # # 4) Compute rewards and aggregate metrics
    #         # rewards = {}
    #         # total_reward = 0.0
    #         for ue in self.ues:
    #             key = f"ue_{ue.id}"
    #             r = self.calculate_individual_reward(key)
    #             rewards[key] = r
    #             total_reward += r                 
            
    #         print(f"Connected Users : {connected_count} Users")
    #         # Log performance metrics at regular intervals
    #         step_time = time.time() - start_time
    #         jains = 0.0
    #         if self.current_step % 10 == 0 or connected_count < self.num_ue:
    #             print(f"Step {self.current_step} | Time: {step_time:.3f}s | Connected: {connected_count}/{self.num_ue} UEs")
                
    #             # Calculate load balancing metrics
    #             loads = [bs.load for bs in self.base_stations]
    #             capacities = [bs.capacity for bs in self.base_stations]
    #             utilizations = [load/cap if cap > 0 else 0 for load, cap in zip(loads, capacities)]
                
    #             # Jain's fairness index for load distribution
    #             if sum(utilizations) > 0:
    #                 squared_sum = sum(utilizations)**2
    #                 sum_squared = sum(u**2 for u in utilizations) * len(utilizations)
    #                 jains = squared_sum / sum_squared if sum_squared > 0 else 0
    #                 print(f"Load Balancing Fairness: {jains:.4f}")
            
    #         # Track current solution for visualization
    #         current_solution = []
    #         for ue in self.ues:
    #             if ue.associated_bs is not None:
    #                 current_solution.append(ue.associated_bs)
    #             else:
    #                 # Use a default value for unconnected UEs
    #                 current_solution.append(-1)  # Using -1 to clearly indicate unconnected status
            
    #         # Calculate SINR values, with protection against overflow
    #         sinr_list = []
    #         for ue in self.ues:
    #             if ue.associated_bs is not None:
    #                 # Ensure SINR is within a reasonable range to prevent overflow
    #                 capped_sinr = min(ue.sinr, 100.0)  # Cap at 100 dB to prevent overflow
    #                 sinr_list.append(capped_sinr)
    #             else:
    #                 sinr_list.append(-np.inf)
            
    #         # KPI logging
    #         if self.log_kpis and self.kpi_logger:
    #             # Safe throughput calculation to prevent overflow
    #             throughput = 0.0
    #             for ue in self.ues:
    #                 if ue.associated_bs is not None:
    #                     # Prevent overflow by capping SINR
    #                     capped_sinr = min(ue.sinr, 100.0)  # Cap at 100 dB
    #                     throughput += np.log2(1 + 10**(capped_sinr/10))
                
    #             metrics = {
    #                 "connected_ratio": connected_count/self.num_ue,
    #                 "step_time": step_time,
    #                 "episode_reward_mean": total_reward/self.num_ue,
    #                 "fairness_index": jains,
    #                 "throughput": throughput,
    #                 "solution": current_solution,
    #                 "sinr_list": sinr_list,
    #             }
    #             self.kpi_logger.log_metrics(
    #                 phase="environment", algorithm="hybrid_marl",
    #                 metrics=metrics, episode=self.current_step
    #             )
                
    #         # Split termination into terminated and truncated (for Gymnasium compatibility)
    #         terminated = {"__all__": False}  # Episode is not terminated due to failure condition
    #         truncated = {"__all__": self.current_step >= self.episode_length}  # Episode length limit reached 

    #         # Common info for all agents
    #         common_info = {
    #             "connected_ratio": connected_count / self.num_ue,
    #             "step_time": step_time,
    #             "current_solution": current_solution,
    #             "avg_reward": total_reward / self.num_ue if self.num_ue > 0 else 0
    #         }
            
    #         # Create info dict with one entry per agent, plus common info
    #         info = {
    #             f"ue_{ue.id}": {
    #                 "connected": ue.associated_bs is not None,
    #                 "sinr": float(min(ue.sinr, 100.0) if ue.associated_bs is not None else -np.inf)  # Cap SINR values
    #             } for ue in self.ues  # Add minimal agent-specific info
    #         }
    #         info["__common__"] = common_info  # Add common info under special key
    #         # Save as last info
    #         self.last_info = info
    #         # Increment step counter
    #         self.current_step += 1
    #         return self._get_obs(), rewards, terminated, truncated, info
        
    #     except Exception as e:
    #         print(f"ERROR in step: {e}")
    #         import traceback
    #         print(traceback.format_exc())
    #         # Return a safe default response
    #         return self._get_obs(), {f"ue_{ue.id}": 0.0 for ue in self.ues}, {"__all__": False}, {"__all__": True}, {"__common__": {"error": str(e)}}   
    # Add this to your NetworkEnvironment class
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
        
        Returns a dictionary of per-UE observations.
        """
        # 1) PRB-based load fraction per BS - with safety check for division
        bs_prb_loads = np.array([bs.load for bs in self.base_stations], dtype=np.float32)
        
        # Safety check to prevent division by zero
        bs_num_rbs = np.array([max(bs.num_rbs, 1) for bs in self.base_stations])
        prb_fractions = bs_prb_loads / bs_num_rbs
        
        # 2) Calculate data-rate utilization per BS with appropriate clipping
        rate_loads = np.array([
            sum(bs.allocated_resources.values()) 
            for bs in self.base_stations
        ], dtype=np.float32)
        
        # Ensure capacity values are valid and reasonable
        cap_bps = np.array([
            max(bs.capacity * 1e6, 1.0)  # Ensure minimum capacity of 1 bps
            for bs in self.base_stations
        ], dtype=np.float32)
        
        # Calculate utilization with safety checks
        # 1. Guard against division by very small numbers
        # 2. Clip the result to prevent extremely large values
        util_bps = np.clip(rate_loads / cap_bps, 0.0, 10.0)  # Cap at 1000% utilization
        
        # Build observations for each UE
        obs = {}
        for ue in self.ues:
            # A) linear SINR to each BS, normalized by its max
            sinr_lin = self._calculate_sinrs(ue).astype(np.float32)
            
            # Add safety check for zero SINR
            max_sinr = max(sinr_lin.max(), 1e-6)  # Minimum value to avoid zero division
            norm_sinr = sinr_lin / max_sinr
            
            # B) own demand normalized to [0,1] with safety check
            max_demand = max(self.max_demand, 1.0)  # Ensure non-zero value
            norm_demand = np.array([min(ue.demand / max_demand, 1.0)], dtype=np.float32)
            
            # C) one-hot for association
            idx = ue.associated_bs if ue.associated_bs is not None else self.num_bs
            one_hot = np.eye(self.num_bs+1, dtype=np.float32)[idx]
            
            # D) concat: SINR | PRB-load | utilization | demand | one-hot
            obs[f"ue_{ue.id}"] = np.concatenate([
                norm_sinr,          # (num_bs,)
                prb_fractions,      # (num_bs,)
                util_bps,           # (num_bs,) - now with safety checks
                norm_demand,        # (1,)
                one_hot             # (num_bs+1,)
            ], axis=0)
        
        return obs

    # def _get_obs(self):
    #     # Precompute BS loads & utils
    #     bs_loads = np.array([bs.load for bs in self.base_stations], dtype=np.float32)
    #     bs_caps  = np.array([bs.capacity for bs in self.base_stations], dtype=np.float32)
    #     normalized_loads = bs_loads / (bs_caps + 1e-9)
    #     bs_utils = np.clip(bs_loads / (bs_caps + 1e-9), 0.0, 1.0).astype(np.float32)

    #     obs = {}
    #     for ue in self.ues:
    #         # 1) SINR vector
    #         sinr_vec = self._calculate_sinrs(ue).astype(np.float32)
    #         normalized_sinrs = sinr_vec / 40.0

    #         # 2) Own demand
    #         norm_demand = np.array([ue.demand / 200.0], dtype=np.float32)

    #         # 3) One-hot current association
    #         #    index = BS id 0…num_bs-1, or num_bs if None
    #         idx = ue.associated_bs if ue.associated_bs is not None else self.num_bs
    #         bs_one_hot = np.eye(self.num_bs + 1, dtype=np.float32)[idx]

    #         # 4) Concatenate everything in the prescribed order
    #         obs[f"ue_{ue.id}"] = np.concatenate([
    #             normalized_sinrs,      # shape (num_bs,)
    #             normalized_loads,      # shape (num_bs,)
    #             bs_utils,              # shape (num_bs,)
    #             norm_demand,           # shape (1,)
    #             bs_one_hot             # shape (num_bs+1,)
    #         ], axis=0)

    #     return obs
   
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
            } for ue in self.ues],
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
        for ue_state in state["users"]:
            ue = next(u for u in self.ues if u.id == ue_state["id"])
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
        for ue in self.ues:
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
        for ue in self.ues:
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
            active_ues = {ue.id: ue for ue in self.ues if ue.associated_bs == bs.id}
            bs.ues = active_ues
            bs.allocate_prbs()    # your PRB-by-PRB PF method

        # --- 6) Track load history & handovers ---
        for bs in self.base_stations:
            self.load_history[bs.id].append(bs.load)
        for ue in self.ues:
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
        # 1) Snapshot current state
        original = self.get_state_snapshot()

        # 2) Apply the proposed associations
        self.apply_solution(solution)

        # 3) Ensure loads and SINRs are up-to-date
        for bs in self.base_stations:
            bs.calculate_load()

        for ue in self.ues:
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
        throughputs_Gbps = throughputs / 1e9  # bits/sec → bytes/sec → Gb/sec
        avg_throughput_Gbps = throughputs_Gbps.mean()

        # # Debug: print a few sample UE stats in GB/s
        # for ue_id in throughputs.argsort()[-5:]:
        #     ue = self.ues[ue_id]            
        #     lin_snr = ue.sinr
        #     snr_db  = 10*np.log10(lin_snr + 1e-12)
        #     r_gbps = throughputs_Gbps[ue_id]
        #     print(f" UE {ue_id}: assoc→BS{ue.associated_bs}, "
        #         f"SINR={snr_db:.2f} dB, Rate={r_gbps:.3f} Gb/s")        

        # 5) Compute other metrics
        fitness     = self.calculate_reward()          # global reward
        avg_sinr    = np.mean([ue.sinr for ue in self.ues])
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
            "average_sinr": float(avg_sinr),
            "average_throughput": float(avg_throughput_Gbps),
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