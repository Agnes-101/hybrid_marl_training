import sys
import os

# Add project root to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


class KPITracker:
    def __init__(self, enabled=True, log_dir="logs", real_time_plot=True):
        """
        Initializes KPI Tracker with logging & live visualization.
        
        :param enabled: Enables/Disables KPI logging.
        :param log_dir: Directory to save logs.
        :param real_time_plot: Enables real-time plotting.
        """
        self.enabled = enabled
        self.real_time_plot = real_time_plot
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.kpi_data = {}
        
        # ✅ Create figure & axis ONCE
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.lines = {}  # Store line objects for each metric-algorithm pair
        plt.ion()  # Enable interactive mode

    def log_kpis(self, episode, avg_reward, avg_sinr, fairness, load_balance, meta_algorithm):
        """
        Logs KPIs for the given episode and updates the live graph if enabled.
        """
        if self.enabled:
            print(f"[Logging] Ep {episode} | Reward: {avg_reward:.2f}, SINR: {avg_sinr:.2f}, "
                  f"Fairness: {fairness:.2f}, Load: {load_balance:.2f} ({meta_algorithm})")

            if meta_algorithm not in self.kpi_data:
                self.kpi_data[meta_algorithm] = []

            self.kpi_data[meta_algorithm].append({
                "episode": episode,
                "reward": avg_reward,
                "sinr": avg_sinr,
                "fairness": fairness,
                "load_balance": load_balance
            })
            
            if self.real_time_plot:
                self.plot_kpis(live_update=True)

    def save_to_csv(self):
        """Saves the KPI logs for all algorithms."""
        for algo, data in self.kpi_data.items():
            df = pd.DataFrame(data)
            filename = os.path.join(self.log_dir, f"kpi_logs_{algo}.csv")
            df.to_csv(filename, index=False)
            print(f"Saved KPI logs for {algo} to {filename}")

    def evaluate_checkpoint(self, checkpoint_path):
        """
        Simulates KPI evaluation for a checkpoint (replace with real evaluation logic).
        """
        import random
        avg_reward = random.uniform(-200, 200)
        avg_sinr = random.uniform(-10, 30)
        fairness = random.uniform(0, 1)
        load_balance = random.uniform(0, 1)
        return avg_reward, avg_sinr, fairness, load_balance

    def plot_kpis(self, live_update=True, final=False):
        """ Updates the existing plot instead of creating new figures """
        if not self.kpi_data:
            return
        
        self.ax.clear()  # ✅ Clear the axis instead of making new figures
        
        for algo, data in self.kpi_data.items():
            df = pd.DataFrame(data)

            # ✅ Check if lines exist; if not, create them
            if algo not in self.lines:
                self.lines[algo] = {
                    "reward": self.ax.plot(df["episode"], df["reward"], label=f"{algo} - Reward", marker="o")[0],
                    "sinr": self.ax.plot(df["episode"], df["sinr"], label=f"{algo} - SINR", linestyle="dashed")[0],
                    "fairness": self.ax.plot(df["episode"], df["fairness"], label=f"{algo} - Fairness", linestyle="dotted")[0],
                    "load_balance": self.ax.plot(df["episode"], df["load_balance"], label=f"{algo} - Load Balance", linestyle="dashdot")[0],
                }
            else:
                # ✅ Update line data instead of replotting
                self.lines[algo]["reward"].set_xdata(df["episode"])
                self.lines[algo]["reward"].set_ydata(df["reward"])
                self.lines[algo]["sinr"].set_xdata(df["episode"])
                self.lines[algo]["sinr"].set_ydata(df["sinr"])
                self.lines[algo]["fairness"].set_xdata(df["episode"])
                self.lines[algo]["fairness"].set_ydata(df["fairness"])
                self.lines[algo]["load_balance"].set_xdata(df["episode"])
                self.lines[algo]["load_balance"].set_ydata(df["load_balance"])

        self.ax.set_xlabel("Episodes")
        self.ax.set_ylabel("KPI Values")
        self.ax.set_title("Final KPI Comparison" if final else "Live KPI Trends")
        self.ax.legend()
        self.ax.grid()

        self.fig.canvas.draw()  # ✅ Force update without blocking execution
        self.fig.canvas.flush_events()  # ✅ Flush GUI events for smooth update