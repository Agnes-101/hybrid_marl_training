import os
import csv
import matplotlib.pyplot as plt

class KPI_Logger:
    def __init__(self, log_dir="logs", enable_tracking=True):
        self.enable_tracking = enable_tracking
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.kpi_file = os.path.join(log_dir, "kpi_log.csv")

        # KPI fields
        self.kpis = ["episode", "SINR", "fairness", "load_balance", "handover_rate"]
        
        # Create a new CSV file if it does not exist
        if not os.path.exists(self.kpi_file):
            with open(self.kpi_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(self.kpis)  # Write header

    def log_kpis(self, episode, sinr, fairness, load_balance, handover_rate):
        """
        Logs the KPIs for each episode.
        """
        if not self.enable_tracking:
            return  # Skip logging if disabled
        
        with open(self.kpi_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([episode, sinr, fairness, load_balance, handover_rate])

    def plot_kpis(self):
        """
        Generates line plots for the KPIs.
        """
        if not os.path.exists(self.kpi_file):
            print("No KPI log found.")
            return

        episodes = []
        sinr_vals = []
        fairness_vals = []
        load_balance_vals = []
        handover_vals = []

        with open(self.kpi_file, mode="r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                episodes.append(int(row[0]))
                sinr_vals.append(float(row[1]))
                fairness_vals.append(float(row[2]))
                load_balance_vals.append(float(row[3]))
                handover_vals.append(float(row[4]))

        # Plot KPI metrics
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 2, 1)
        plt.plot(episodes, sinr_vals, label="SINR", color="b")
        plt.xlabel("Episodes")
        plt.ylabel("SINR (dB)")
        plt.title("SINR Over Time")
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.plot(episodes, fairness_vals, label="Fairness", color="g")
        plt.xlabel("Episodes")
        plt.ylabel("Fairness Index")
        plt.title("Fairness Over Time")
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.plot(episodes, load_balance_vals, label="Load Balance", color="r")
        plt.xlabel("Episodes")
        plt.ylabel("Load Balance")
        plt.title("Load Balance Over Time")
        plt.grid()

        plt.subplot(2, 2, 4)
        plt.plot(episodes, handover_vals, label="Handover Rate", color="m")
        plt.xlabel("Episodes")
        plt.ylabel("Handover Rate")
        plt.title("Handover Rate Over Time")
        plt.grid()

        plt.tight_layout()
        plt.show()
