        
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
                ewma_alpha=0.9):
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
        self.ewma_dr       = 0.0
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
                    
class BaseStation:
    def __init__(self, id, position, frequency, bandwidth, height=50.0, reuse_color=None,tx_power_dbm=30.0,tx_gain_dbi=8.0,bf_gain_dbi=20.0):
        self.id = int(id)
        self.position = np.array(position, dtype=np.float32)
        self.frequency = float(frequency)    # MHz
        self.bandwidth = float(bandwidth)    # MHz
        self.height = float(height)          # m
        self.tx_power = float(tx_power_dbm)  # dBm
        self.tx_gain       = float(tx_gain_dbi)  # dBi
        self.bf_gain      = bf_gain_dbi   # beamforming gain
        self.allocated_resources = {}        # {ue_id: allocated_rate}
        self.load = 0.0                      # sum of allocated rates
        self.capacity = self.bandwidth * 0.8 # e.g. 80% of bandwidth in Mbps
        self.reuse_color   = reuse_color         # e.g. "A", "B", or "C"
        
    def calculate_load(self):
        self.load = sum(self.allocated_resources.values())

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
            
    # def received_power_mW(self, ue_pos, ue_rx_gain=None):
    #     d = np.linalg.norm(self.position - ue_pos)
    #     # path loss in dB
    #     L = self.path_loss(d, d0=1.0, n=2.5, sigma=8.0)
    #     # total gain in dB
    #     G_tx = self.tx_gain
    #     G_rx = ue_rx_gain if ue_rx_gain is not None else 0.0
    #     # link budget: P_rx (dBm) = P_tx(dBm) + G_tx + G_rx - L
    #     p_rx_dbm = self.tx_power + G_tx + G_rx - L
    #     # convert to mW
    #     return 10 ** (p_rx_dbm / 10)
    
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
    
    def data_rate_shared(self, ue, alpha=1.0, beta=1.0):
        # Ensure UE is “connected” so it appears in allocated_resources
        if ue.id not in self.allocated_resources:
            self.allocated_resources[ue.id] = 0.0

        # 1) Compute priority for every UE on this BS
        weights = []
        for other_ue_id in self.allocated_resources:
            other_ue = self.ues[other_ue_id]
            w = self.priority(other_ue, alpha, beta)
            weights.append(w)

        W = sum(weights) + 1e-9

        # 2) Fraction for THIS UE
        w_i = self.priority(ue, alpha, beta)
        fraction = w_i / W

        # 3) Shared rate = fraction × instantaneous capacity
        T_inst = self.data_rate_unshared(ue)
        return fraction * T_inst

class NetworkEnvironment(MultiAgentEnv):
    @staticmethod
    def generate_hex_positions(num_bs, width=100.0, height=100.0):
        """
        Return up to num_bs (row, col, [x,y]) tuples on a hex lattice.
        """
        n_cols = int(math.ceil(math.sqrt(num_bs * width / height)))
        n_rows = int(math.ceil(num_bs / n_cols))

        # horizontal & vertical pitch
        pitch_x = width / (n_cols + 0.5)
        pitch_y = pitch_x * math.sin(math.radians(60))
        if pitch_y * (n_rows + 0.5) > height:
            pitch_y = height / (n_rows + 0.5)
            pitch_x = pitch_y / math.sin(math.radians(60))

        grid = []
        for row in range(n_rows):
            y = pitch_y * (row + 0.5)
            x_offset = 0.5 * pitch_x if (row % 2 == 1) else 0.0
            for col in range(n_cols):
                x = pitch_x * col + x_offset + 0.5 * pitch_x
                if x <= width - 0.5 * pitch_x and y <= height - 0.5 * pitch_y:
                    grid.append((row, col, [x, y]))
                    if len(grid) == num_bs:
                        return grid
        return grid[:num_bs]
    
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
        for i, (row, col, pos) in enumerate(grid):
            color = colors[(row + 2*col) % len(colors)]
            freq  = carrier_freqs[color]
            bs = BaseStation(
                id=i,
                position=pos,
                frequency=freq,
                bandwidth=bandwidth,
                reuse_color=color
            )
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
        for bs in self.base_stations:
            bs.base_stations = self.base_stations  # for interference loops
            bs.ues           = self.ues            # so data_rate_shared can see all UEs
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
    def reset(self, seed=None, options=None):
        self.current_step = 0

        # 1) Clear BS allocations & loads
        for bs in self.base_stations:
            bs.allocated_resources.clear()
            bs.load = 0.0

        # 2) Reset UEs
        for ue in self.ues:
            ue.position      = np.random.uniform(0,100,2).astype(np.float32)
            ue.associated_bs = None
            ue.sinr          = -np.inf
            ue.ewma_dr       = 0.0

        # 3) (Optional) reset any KPI histories
        self.prev_associations = {ue.id: None for ue in self.ues}
        self.handover_counts  = {ue.id: 0    for ue in self.ues}
        self.step_count       = 0
        if hasattr(self, "load_history"):
            for bs in self.base_stations:
                self.load_history[bs.id].clear()

        return self._get_obs(), {}

    # Add these to your NetworkEnvironment class
    def calculate_jains_fairness(self):
        throughputs = [ue.throughput for ue in self.ues]
        return (sum(throughputs) ** 2) / (len(throughputs) * sum(t**2 for t in throughputs))

    @property
    def throughput(self):
        return torch.log2(1 + 10**(self.sinr/10)).item()
        
    def calculate_reward(self):
        """Normalized reward: Gbps throughput + fairness - overload_penalty."""
        # 1) Sum actual per-UE shared throughput in Gbps
        #    (we assume bs.allocated_resources[u.id] is in bps)
        total_bps = sum(
            dr for bs in self.base_stations for dr in bs.allocated_resources.values()
        )
        throughput_gbps = total_bps / 1e9

        # 2) Normalize each BS load by its capacity (bps)
        loads_norm = torch.tensor(
            [bs.load / bs.capacity for bs in self.base_stations],
            dtype=torch.float32
        )
        
        # 3) Jain fairness on these fractions
        jain = (loads_norm.sum()**2) / (self.num_bs * (loads_norm**2).sum() + 1e-6)

        # 4) Overload penalty: how many cells exceed 100% load
        overload = torch.relu(loads_norm - 1.0).sum()  # unitless

        # 5) Combine into a single reward
        #    weights chosen so throughput (Gbps) is comparable to jain (0–1) and overload
        return (
            1.0 * throughput_gbps     # reward raw capacity in Gbps
        + 2.0 * jain                # reward fairness [0–2]
        - 1.0 * overload           # penalty per “overloaded cell” fraction
        )

    
    def pick_best_bs(self, ue, w1=0.7, w2=0.3, delta=0.1):
        """
        Returns the best BS index for UE, but only hands over if
        new_score >= old_score + delta.
        
        - w1, w2: weighting for rate vs. load
        - delta: minimum score improvement to trigger handover
        """
        # 1) Compute normalized unshared rates
        unshared = np.array([bs.data_rate_unshared(ue) for bs in self.base_stations])
        max_unsh = unshared.max() + 1e-9
        rate_score = unshared / max_unsh

        # 2) Compute load factor score
        loads = np.array([bs.load for bs in self.base_stations])
        caps  = np.array([bs.capacity for bs in self.base_stations])
        load_score = 1 - loads / caps

        # 3) Composite score vector
        scores = w1 * rate_score + w2 * load_score

        # 4) Find candidate
        cand_idx   = int(np.argmax(scores))
        cand_score = scores[cand_idx]

        # 5) If already associated, check threshold
        if ue.associated_bs is not None:
            cur_idx   = ue.associated_bs
            cur_score = scores[cur_idx]
            # only switch if cand_score >= cur_score + delta
            if cand_score >= cur_score + delta:
                return cand_idx
            else:
                return cur_idx

        # 6) If not yet associated, just pick the best
        return cand_idx

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
        Return an array of linear SINRs from every BS → ue link.
        If debug=True, print the first few link budgets in dB.
        """
        sinrs = []
        for bs in self.base_stations:
            # signal + interference + noise (all in mW)
            prx   = bs.received_power_mW(ue.position, ue.rx_gain)
            interf = sum(
                other.received_power_mW(ue.position, ue.rx_gain)
                for other in self.base_stations
                if (other.id != bs.id) and (other.reuse_color == bs.reuse_color)
            )
            noise = bs.noise_mW()

            lin_snr = prx / (interf + noise + 1e-12)
            snr_db  = 10 * np.log10(lin_snr)

            if debug and ue.id < 3:
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
    
    
    def step(self, actions: Dict[str, int]):
        # Add timing for performance analysis
        print(f"Step called with {len(actions)} actions")
        try:
            start_time = time.time()

            # 1) Clear old allocations & reset loads
            for bs in self.base_stations:
                bs.allocated_resources.clear()
                bs.calculate_load()

            # 2) Process UE associations (sorted by demand for packing)
            ue_actions = sorted(
                ((int(aid.split("_")[1]), bs_choice) for aid, bs_choice in actions.items()),
                key=lambda x: self.ues[x[0]].demand
            )
            bs_allocations = {bs.id: 0 for bs in self.base_stations}
            bs_capacity_used = {bs.id: 0.0 for bs in self.base_stations}
            connected_count = 0
            
            # assume ue_actions is a list of (ue_id, _) sorted by demand
            for ue_id, _ in ue_actions:
                ue = self.ues[ue_id]

                # 1) primary choice by PF+load score
                primary = self.pick_best_bs(ue)

                # 2) build ordered list: primary first, then the rest
                candidates = [primary] + [i for i in range(self.num_bs) if i != primary]

                # 3) try each candidate until one fits
                admitted = False
                for bs_idx in candidates:
                    bs = self.base_stations[bs_idx]
                    dr = bs.data_rate_shared(ue)               # how much rate UE would use
                    if bs.load + dr <= bs.capacity:            # capacity check
                        bs.allocated_resources[ue.id] = dr
                        bs.calculate_load()                    # update bs.load
                        ue.associated_bs = bs_idx
                        connected_count += 1                # ← increment here
                        bs_allocations[bs_idx] += 1        # count UEs per BS
                        bs_capacity_used[bs_idx] = bs.load # record load
                        admitted = True
                        break

                # 4) if none could admit, leave unassociated
                if not admitted:
                    ue.associated_bs = None

            
            # 3) Update SINR & EWMA
            for ue in self.ues:
                # full multi-cell SINR vector
                sinr_vec = self._calculate_sinrs(ue)
                if ue.associated_bs is not None:
                    ue.sinr = sinr_vec[ue.associated_bs]
                    # update EWMA with allocated rate
                    self.base_stations[ue.associated_bs].calculate_load()
                    measured = self.base_stations[ue.associated_bs].allocated_resources.get(ue.id, 0.0)
                    ue.update_ewma(measured)
                else:
                    ue.sinr = -np.inf
            
            # Calculate rewards efficiently
            rewards = {}
            total_reward = 0
            for bs in self.base_stations:
                print(f"BS {bs.id}: load={bs.load}, capacity={bs.capacity}, ratio={bs.load/bs.capacity if bs.capacity>0 else 'inf'}")
            # 4) Compute rewards and aggregate metrics
            rewards = {}
            total_reward = 0.0
            for ue in self.ues:
                key = f"ue_{ue.id}"
                r = self.calculate_individual_reward(key)
                rewards[key] = r
                total_reward += r                 
                
                # if ue_id % 10 == 0:  # Only print every 10th UE to avoid excessive output
                #         print(f"UE {ue_id}: connected={ue.associated_bs is not None}, "
                #             f"sinr={ue.sinr:.2f}, throughput={throughput:.2f}, "
                #             f"load_factor={load_factor:.2f}, reward={rewards[f'ue_{ue.id}']:.4f}")
                        
            print(f"Connected Users : {connected_count} Users")
            # Log performance metrics at regular intervals
            step_time = time.time() - start_time
            jains = 0.0
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
                    jains = squared_sum / sum_squared if sum_squared > 0 else 0
                    print(f"Load Balancing Fairness: {jains:.4f}")
                    # jain = self.calculate_jains_fairness()
                    # print(f"Load balancing Jain's index: {jain:.4f}")
                
                # # Log detailed BS allocations if not all UEs connected
                # if connected_count < self.num_ue:
                #     print(f"BS allocations: {bs_allocations}")
                #     print(f"BS capacity %: {[bs_capacity_used[bs.id]/bs.capacity if bs.capacity > 0 else 0 for bs in self.base_stations]}")
            
            
            
            # Track current solution for visualization
            current_solution = []
            for ue in self.ues:
                if ue.associated_bs is not None:
                    current_solution.append(ue.associated_bs)
                else:
                    # Use a default value for unconnected UEs
                    current_solution.append(0)  # Or None if your visualization can handle it
            # print(f"Current solution at : {current_solution}")
            sinr_list = [ue.sinr if ue.associated_bs is not None else -np.inf
             for ue in self.ues]
            
            # KPI logging
            if self.log_kpis and self.kpi_logger:
                metrics = {
                    "connected_ratio": connected_count/self.num_ue,
                    "step_time": step_time,
                    "episode_reward_mean": total_reward/self.num_ue,
                    "fairness_index": jains,
                    "throughput": sum(np.log2(1+10**(ue.sinr/10)) for ue in self.ues if ue.associated_bs is not None),
                    "solution":       current_solution,
                    "sinr_list":      sinr_list,
                }
                self.kpi_logger.log_metrics(
                    phase="environment", algorithm="hybrid_marl",
                    metrics=metrics, episode=self.current_step
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
            # Increment step counter
            self.current_step += 1
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
        # Precompute BS loads, capacities, positions
        bs_loads = np.array([bs.load for bs in self.base_stations], dtype=np.float32)
        bs_caps  = np.array([bs.capacity for bs in self.base_stations], dtype=np.float32)
        bs_positions = np.stack([bs.position for bs in self.base_stations])

        # Normalize loads by capacity
        normalized_loads = bs_loads / (bs_caps + 1e-9)
        # Utilization = clipped load/capacity
        bs_utils = np.clip(bs_loads / (bs_caps + 1e-9), 0.0, 1.0).astype(np.float32)

        obs = {}
        for ue in self.ues:
            # Compute full SINR vector (interference + noise)
            sinr_vec = self._calculate_sinrs(ue)
            # Record UE.sinr
            ue.sinr = sinr_vec[ue.associated_bs] if ue.associated_bs is not None else -np.inf

            # Normalize SINRs (assume max ~40 dB)
            normalized_sinrs = sinr_vec / 40.0

            # Normalize own demand
            norm_demand = np.array([ue.demand / 200.0], dtype=np.float32)

            # Concatenate into one feature vector
            obs[f"ue_{ue.id}"] = np.concatenate([
                normalized_sinrs.astype(np.float32),
                normalized_loads,
                bs_utils,
                norm_demand
            ], axis=0)

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
    
    def get_current_state(self):
        return {
            "base_stations": [
                {
                    "id": bs.id,
                    "position": bs.position.tolist(),
                    "load": bs.load,
                    "capacity": bs.capacity,
                    "reuse_color": bs.reuse_color,     # new
                    "frequency": bs.frequency          # new
                } for bs in self.base_stations
            ],
            "users": [
                {
                    "id": ue.id,
                    "position": ue.position.tolist(),
                    "associated_bs": ue.associated_bs,
                    "sinr": float(ue.sinr),
                    "speed": float(ue.speed),           # new
                    "pause_time": float(ue.pause_time), # new
                    "ewma_dr": float(ue.ewma_dr)        # optional
                } for ue in self.ues
            ],
            "associations": self.associations.copy(),
            "version": self.version,
        }

        
    def get_state_snapshot(self) -> dict:
        return {
            "users": [{
                "id": ue.id,
                "position": ue.position.copy().tolist(),
                # MRWP fields instead of raw velocity:
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
                "load": float(bs.load),
                "capacity": float(bs.capacity),
                # if you care about reuse_color/frequency snapshot:
                "reuse_color": bs.reuse_color,
                "frequency": float(bs.frequency)
            } for bs in self.base_stations],
            "current_step": self.current_step
        }

    def set_state_snapshot(self, state: dict):
        # restore UEs
        for ue_state in state["users"]:
            ue = next(u for u in self.ues if u.id == ue_state["id"])
            ue.position     = np.array(ue_state["position"], dtype=np.float32)
            ue.waypoint     = np.array(ue_state["waypoint"], dtype=np.float32)
            ue.speed        = ue_state["speed"]
            ue.pause_time   = ue_state["pause_time"]
            ue.demand       = ue_state["demand"]
            ue.associated_bs = ue_state["associated_bs"]
            ue.sinr         = ue_state["sinr"]
            ue.ewma_dr      = ue_state["ewma_dr"]
        # restore BSs
        for bs_state in state["base_stations"]:
            bs = next(b for b in self.base_stations if b.id == bs_state["id"])
            bs.allocated_resources = bs_state["allocated_resources"].copy()
            bs.load               = bs_state["load"]
            bs.capacity           = bs_state["capacity"]
            # if you snapshot these too:
            bs.reuse_color        = bs_state.get("reuse_color", bs.reuse_color)
            bs.frequency          = bs_state.get("frequency", bs.frequency)
        self.current_step = state["current_step"]


    
    def _update_system_metrics(self):
        # 1) Recompute loads from PF‐shared allocations
        for bs in self.base_stations:
            bs.calculate_load()

        # 2) Recompute each UE’s SINR using the multi‐cell helper
        for ue in self.ues:
            if ue.associated_bs is not None:
                sinr_vec = self._calculate_sinrs(ue)     # NumPy vector of SINRs
                ue.sinr = sinr_vec[ue.associated_bs]
            else:
                ue.sinr = -np.inf

    
    def apply_solution(self, solution):
        """Apply a solution (numpy array or dict) to the environment"""
        # 1) Unwrap array into {bs_id:[ue_ids]} if needed
        if isinstance(solution, np.ndarray):
            sol_dict = {bs.id: [] for bs in self.base_stations}
            for ue_idx, bs_id in enumerate(solution.astype(int)):
                sol_dict[int(bs_id)].append(ue_idx)
            solution = sol_dict

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
            # bs.allocated_resources = {}
            bs.allocated_resources.clear()
            bs.calculate_load()
            self.associations[bs.id] = []

        # 4) Assign PF‐shared rates and mark UEs as associated
        for bs_id, ue_list in solution.items():
            bs = next(b for b in self.base_stations if b.id == int(bs_id))
            for ue_id in ue_list:
                ue = self.ues[ue_id]
                # 4a) compute and store the shared rate
                dr = bs.data_rate_shared(ue)
                bs.allocated_resources[ue.id] = dr
                # 4b) set the UE’s associated_bs
                ue.associated_bs = bs.id
                # 4c) record in your associations dict
                self.associations[bs.id].append(ue.id)
            bs.calculate_load()
        for bs in self.base_stations:
            self.load_history[bs.id].append(bs.load)
        for ue in self.ues:
            old = self.prev_associations[ue.id]
            new = ue.associated_bs
            if (old is not None) and (new is not None) and (old != new):
                self.handover_counts[ue.id] += 1
            self.prev_associations[ue.id] = new
        self.step_count += 1
        # 5) Refresh SINR metrics
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
            "average_throughput_Gbps": float(avg_throughput_Gbps),
            "fairness": float(fairness),
            "load_variance": float(load_var),
            "bs_loads": bs_loads,
            "handover_rate": ho_rate_per_step,
            "load_quantiles_Gbps": {"10th": q10, "50th": q50, "90th": q90}

        }

    # def evaluate_detailed_solution(self, solution, alpha=0.1, beta=0.1):
    #     """
    #     Apply a candidate user‑association solution, compute detailed performance metrics, then restore state.
    #     """
    #     # 1) Snapshot current state
    #     original = self.get_state_snapshot()

    #     # 2) Apply the proposed associations
    #     self.apply_solution(solution)
    #     connected_count = sum(1 for ue in self.ues if ue.associated_bs is not None)
    #     # print(f"Connected UEs = {connected_count} / {self.num_ue}")
        
    #     # 3) Ensure loads and SINRs are up-to-date
    #     for bs in self.base_stations:
    #         bs.calculate_load()
    #     # for ue in self.ues:
    #     #     ue.sinr = self._calculate_sinrs(ue)[ue.associated_bs] if ue.associated_bs is not None else -np.inf
    #     for ue in self.ues:
    #         if ue.associated_bs is not None:
    #             sinr_vec = self._calculate_sinrs(ue, debug=(ue.id<3))
    #             ue.sinr = sinr_vec[ue.associated_bs]
    #         else:
    #             ue.sinr = -np.inf
    #     # 4) Throughputs per UE from actual shared allocations
    #     throughputs = np.zeros(self.num_ue, dtype=np.float32)
    #     for bs in self.base_stations:
    #         for ue_id, dr in bs.allocated_resources.items():
    #             throughputs[ue_id] = dr

    #     # Debug: print a few sample UE stats
    #     for ue_id in throughputs.argsort()[-5:]:
    #         ue = self.ues[ue_id]
    #         print(f" UE {ue_id}: assoc→BS{ue.associated_bs}, SINR={ue.sinr:.2f} dB, Rate={throughputs[ue_id]:.3f} Mbps")

    #     avg_throughput = throughputs.mean()

    #     # 5) Compute metrics
    #     fitness = self.calculate_reward()  # global environment reward
    #     avg_sinr = np.mean([ue.sinr for ue in self.ues])        
    #     fairness = (throughputs.sum()**2) / (len(throughputs) * np.sum(throughputs**2) + 1e-9)
    #     load_var = np.var([bs.load for bs in self.base_stations])
    #     bs_loads = [bs.load for bs in self.base_stations]

    #     # 6) Restore original state
    #     self.set_state_snapshot(original)

    #     # 7) Return detailed report
    #     return {
    #         "fitness": float(fitness),
    #         "average_sinr": float(avg_sinr),
    #         "average_throughput": float(avg_throughput),
    #         "fairness": float(fairness),
    #         "load_variance": float(load_var),
    #         "bs_loads": bs_loads
    #     }


        
# env = NetworkEnvironment({"num_ue": 3, "num_bs": 2})
# obs, _ = env.reset()
# print(obs["ue_0"].shape)  # Should be (2*2 + 1)=5

# actions = {"ue_0": 1, "ue_1": 0, "ue_2": 1}  # Each UE selects a BS index
# next_obs, rewards, dones, _ = env.step(actions)
# print(next_obs, rewards, dones, _ )