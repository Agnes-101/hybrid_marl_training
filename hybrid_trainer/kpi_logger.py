import sys
import os

# Path configuration remains
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

# hybrid_trainer/kpi_logger.py
import pandas as pd
import numpy as np
import time

class KPITracker:
    def __init__(self, enabled=True):        
        self.enabled = enabled
        self.history = pd.DataFrame(columns=[            
            'timestamp', 'episode', 'phase', 'algorithm',
    'episode_reward_mean', 'policy_entropy', 'fitness',
    'average_sinr', 'fairness', 'load_variance', 'diversity'
        ])
        self.logs = []  # Initialize logs storage
        self.data = []  # Initialize data storage
        self.algorithm_logs = []  # New: Store algorithm performance data
        
    def log_metrics(self, episode: int, phase: str, algorithm: str, metrics: dict):
        """Unified logging method"""
        if not self.enabled:
            return
        
        print(f"Logging metrics at episode {episode}: {metrics}", flush=True)
        
        self.history = pd.concat([
            self.history,
            pd.DataFrame([{
                'timestamp': time.time(),
                'episode': episode,
                'phase': phase,
                'algorithm': algorithm,
                'episode_reward_mean': metrics.get('episode_reward_mean', 0),
                'policy entropy': metrics.get ('policy entropy',0),
                'fitness': metrics.get('fitness', 0),
                'average_sinr': metrics.get('average_sinr', 0),
                'fairness': metrics.get('fairness', 0),
                'load_variance': metrics.get('load_variance', 0),
                'diversity':metrics.get('diversity', 0)
            }])
        ], ignore_index=True)
        
        # self.logs.append({
        #         "episode": episode,
        #         "reward": reward,
        #         "sinr": sinr,
        #         "fairness": fairness,
        #         "load_variance": load_variance
        #     })
    
    # def log_metrics(self, timestamp: float, phase: str, 
    #             algorithm: str, metrics: dict):
    #     """Log metrics for a single timestep"""
    #     if not self.enabled:
    #         return
            
    #     new_row = {
    #         'timestamp': timestamp,
    #         'phase': phase,
    #         'algorithm': algorithm,
    #         'reward': metrics.get('reward', 0),
    #         'average_sinr': metrics.get('average_sinr', 0),
    #         'fairness': metrics.get('fairness', 0),
    #         'load_variance': metrics.get('load_variance', 0)
    #     }
        
    #     self.history = pd.concat([
    #         self.history,
    #         pd.DataFrame([new_row])
    #     ], ignore_index=True)
    
    def recent_logs(self, n=5):
        """Retrieve the last n log entries for debugging."""
        return self.logs[-n:]
    
    def log_algorithm_performance(self, algorithm: str, metrics: dict):
        """Log metaheuristic algorithm performance metrics"""
        if self.enabled:
            self.algorithm_logs.append({
                "algorithm": algorithm,
                "timestamp": time.time(),
                **metrics  # Unpack fitness, SINR, fairness, etc.
            })
    
    def get_recent_metrics(self, window_size: int = 100) -> dict:
        """Get metrics for visualization updates"""
        recent = self.history.tail(window_size)
        return {
            'timestamps': recent['timestamp'].tolist(),
            'rewards': recent['reward'].tolist(),
            'sinr': recent['sinr'].tolist(),
            'fairness': recent['fairness'].tolist(),
            'load_variance': recent['load_variance'].tolist(),
            'phases': recent['phase'].tolist(),
            'algorithms': recent['algorithm'].tolist()
        }

    def save_to_csv(self, path: str = "results/kpis.csv"):
        """Persist full history to disk"""
        if self.enabled:
            self.history.to_csv(path, index=False)

    def get_algorithm_comparison(self) -> dict:
        """Aggregate metrics for algorithm comparison view"""
        if not self.enabled:
            return {}
            
        return self.history.groupby('algorithm').agg({
            'reward': ['mean', 'max'],
            'sinr': 'mean',
            'fairness': 'mean',
            'load_variance': 'min'
        }).to_dict()
        
    def log_kpis(self, episode: int, reward: float, sinr: float, 
                fairness: float, load_variance: float):
        if self.enabled:
            # Ensure fairness is scalar
            if isinstance(fairness, (list, np.ndarray)):
                fairness = np.mean(fairness)
            
            self.data.append({
                "episode": episode,
                "reward": float(reward),
                "sinr": float(np.mean(sinr)),
                "fairness": float(fairness),
                "load_variance": float(load_variance)
            })
            
            self.logs.append({
                "episode": episode,
                "reward": reward,
                "sinr": sinr,
                "fairness": fairness,
                "load_variance": load_variance
            })
            
    def generate_final_reports(self):
        """Save logs and plots (minimal example)"""
        df = pd.DataFrame(self.algorithm_logs)
        df.to_csv("results/algorithm_performance.csv", index=False)


# import sys
# import os

# # Add project root to Python's path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, project_root) if project_root not in sys.path else None

# import pandas as pd
# import matplotlib.pyplot as plt
# from IPython.display import display, clear_output


# class KPITracker:
#     def __init__(self, enabled=True, log_dir="logs", real_time_plot=True):
#         """
#         Initializes KPI Tracker with logging & live visualization.
        
#         :param enabled: Enables/Disables KPI logging.
#         :param log_dir: Directory to save logs.
#         :param real_time_plot: Enables real-time plotting.
#         """
#         self.enabled = enabled
#         self.real_time_plot = real_time_plot
#         self.log_dir = log_dir
#         os.makedirs(self.log_dir, exist_ok=True)
#         self.kpi_data = {}
        
#         # ✅ Create figure & axis ONCE
#         self.fig, self.ax = plt.subplots(figsize=(12, 6))
#         self.lines = {}  # Store line objects for each metric-algorithm pair
#         plt.ion()  # Enable interactive mode

#     def log_kpis(self, episode, avg_reward, avg_sinr, fairness, load_balance, meta_algorithm):
#         """
#         Logs KPIs for the given episode and updates the live graph if enabled.
#         """
#         if self.enabled:
#             print(f"[Logging] Ep {episode} | Reward: {avg_reward:.2f}, SINR: {avg_sinr:.2f}, "
#                   f"Fairness: {fairness:.2f}, Load: {load_balance:.2f} ({meta_algorithm})")

#             if meta_algorithm not in self.kpi_data:
#                 self.kpi_data[meta_algorithm] = []

#             self.kpi_data[meta_algorithm].append({
#                 "episode": episode,
#                 "reward": avg_reward,
#                 "sinr": avg_sinr,
#                 "fairness": fairness,
#                 "load_balance": load_balance
#             })
            
#             if self.real_time_plot:
#                 self.plot_kpis(live_update=True)

#     def save_to_csv(self):
#         """Saves the KPI logs for all algorithms."""
#         for algo, data in self.kpi_data.items():
#             df = pd.DataFrame(data)
#             filename = os.path.join(self.log_dir, f"kpi_logs_{algo}.csv")
#             df.to_csv(filename, index=False)
#             print(f"Saved KPI logs for {algo} to {filename}")

#     def evaluate_checkpoint(self, checkpoint_path):
#         """
#         Simulates KPI evaluation for a checkpoint (replace with real evaluation logic).
#         """
#         import random
#         avg_reward = random.uniform(-200, 200)
#         avg_sinr = random.uniform(-10, 30)
#         fairness = random.uniform(0, 1)
#         load_balance = random.uniform(0, 1)
#         return avg_reward, avg_sinr, fairness, load_balance

#     def plot_kpis(self, live_update=True, final=False):
#         """Ensures real-time plot updates continuously."""
#         if not self.kpi_data or not self.real_time_plot:
#             return

#         for algo, data in self.kpi_data.items():
#             df = pd.DataFrame(data)
#             if df.empty:
#                 continue

#             for metric in ['reward', 'sinr', 'fairness', 'load_balance']:
#                 if metric not in self.lines[algo]:
#                     self.lines[algo][metric], = self.ax.plot(
#                         df["episode"], df[metric],
#                         label=f"{algo} - {metric}",
#                         marker="o" if metric == 'reward' else None,
#                         linestyle=self._get_linestyle(metric)
#                     )
#                 else:
#                     self.lines[algo][metric].set_xdata(df["episode"])
#                     self.lines[algo][metric].set_ydata(df[metric])

#         # ✅ Proper axes updates
#         self.ax.relim()
#         self.ax.autoscale_view()

#         # ✅ Ensure smooth GUI updates
#         self._update_legend()
#         self.fig.canvas.draw_idle()
#         plt.pause(0.01)  # Ensure updates are processed

#         if final:  
#             plt.ioff()  # Disable interactive mode if finalizing plot
