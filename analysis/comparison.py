import sys
import os

# Configure project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None
# analysis/comparison.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from IPython.display import HTML
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

# class MetricAnimator:
#     def __init__(self, df: pd.DataFrame, metrics: list, fps: int = 5):
#         self.df = df
#         self.metrics = metrics
#         self.fps = fps
#         self.max_iter = df['episode'].max()
#         self.algorithms = df['algorithm'].unique()
        
#         # Create one figure per metric
#         self.figures = [plt.figure(figsize=(12, 6)) for _ in metrics]
#         self.axes = [fig.add_subplot(111) for fig in self.figures]
#         self.lines = {metric: {} for metric in metrics}

#         # Initialize plots
#         for idx, metric in enumerate(metrics):
#             ax = self.axes[idx]
#             ax.set_title(metric.replace('_', ' ').title())
#             ax.set_xlim(0, self.max_iter)
#             ax.set_ylim(
#                 df[metric].min() * 0.95, 
#                 df[metric].max() * 1.05
#             )
#             ax.grid(True)
            
#             # Create lines for each algorithm
#             for algo in self.algorithms:
#                 self.lines[metric][algo], = ax.plot(
#                     [], [], 
#                     label=algo, 
#                     marker='o',
#                     markersize=4,
#                     markevery=5
#                 )
#             ax.legend()

#     def _update(self, frame: int):
#         """Update all metrics sequentially up to current frame"""
#         current_data = self.df[self.df['episode'] <= frame]
        
#         for metric in self.metrics:
#             for algo in self.algorithms:
#                 algo_data = current_data[
#                     (current_data['algorithm'] == algo)
#                 ].sort_values('episode')
                
#                 self.lines[metric][algo].set_data(
#                     algo_data['episode'], 
#                     algo_data[metric]
#                 )
        
#         return [line for metric in self.metrics 
#                 for line in self.lines[metric].values()]

#     def animate(self):
#         """Create unified animation across all metrics"""
#         self.ani = animation.FuncAnimation(
#             self.figures[0],  # Anchor animation to first figure
#             self._update,
#             frames=range(self.max_iter + 1),
#             interval=1000//self.fps,
#             blit=True
#         )
    
#     def save_videos(self, path: str = "metric_progression.mp4"):
#         """Render to video file"""
#         self.ani.save(path, writer='ffmpeg', fps=self.fps)

#     def show(self):
#         """Display all figures"""
#         plt.show()
        
class MetricAnimator:
    def __init__(self, df: pd.DataFrame, metrics: list, fps: int = 10):
        self.df = df
        self.metrics = metrics
        self.fps = fps
        self.max_iter = df['episode'].max()
        self.algorithms = df['algorithm'].unique()
        
        # Store individual figures and animators
        self.figures = []
        self.animators = []

    def _setup_metric_figure(self, metric: str):
        """Create separate figure for each metric"""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel("Iteration")
        ax.set_ylabel(metric)
        ax.grid(True)
        
        lines = {}
        for algo, color in zip(self.algorithms, plt.cm.tab10.colors):
            line, = ax.plot([], [], label=algo, color=color)
            lines[algo] = line
        
        ax.legend()
        return fig, ax, lines
    def animate(self):
        """Create separate animations per metric"""
        for metric in self.metrics:
            fig, ax, lines = self._setup_metric_figure(metric)
            
            # Capture the current metric in the closure
            def update(frame, current_metric=metric):  # Fix here
                current_data = self.df[self.df['episode'] <= frame]
                for algo in self.algorithms:
                    algo_data = current_data[
                        (current_data['algorithm'] == algo)
                    ].sort_values('episode')
                    # Use current_metric instead of metric
                    lines[algo].set_data(algo_data['episode'], algo_data[current_metric])
                
                ax.relim()
                ax.autoscale_view()
                return list(lines.values())
            
            ani = animation.FuncAnimation(
                fig, update, frames=range(self.max_iter + 1),
                interval=2000//self.fps, blit=True
            )
            self.figures.append(fig)
            self.animators.append(ani)

    # def animate(self):
    #     """Create separate animations per metric"""
    #     for metric in self.metrics:
    #         fig, ax, lines = self._setup_metric_figure(metric)
            
    #         def update(frame):
    #             current_data = self.df[self.df['episode'] <= frame]
    #             for algo in self.algorithms:
    #                 algo_data = current_data[
    #                     (current_data['algorithm'] == algo)
    #                 ].sort_values('episode')
    #                 lines[algo].set_data(algo_data['episode'], algo_data[metric])
                
    #             ax.relim()
    #             ax.autoscale_view()
    #             return list(lines.values())
            
    #         ani = animation.FuncAnimation(
    #             fig, update, frames=range(self.max_iter + 1),
    #             interval=1000//self.fps, blit=True
    #         )
    #         self.figures.append(fig)
    #         self.animators.append(ani)

    def save_videos(self, base_path: str = "results/metrics"):
        """Save one video per metric"""
        os.makedirs(base_path, exist_ok=True)
        for idx, (metric, ani) in enumerate(zip(self.metrics, self.animators)):
            path = f"{base_path}/{metric}_progression.mp4"
            ani.save(path, writer='ffmpeg', fps=self.fps)
            print(f"Saved {path}")

    def show(self):
        """Display all metrics in separate windows"""
        # plt.show(block=False)
        """Render all animations inline in Colab"""
        for ani in self.animators:
            plt.close(ani._fig)  # Avoid duplicate figures
            display(HTML(ani.to_jshtml()))



# class MetricAnimator:
#     def __init__(self, df: pd.DataFrame, metrics: list, fps: int = 10):
#         """
#         df: Consolidated DataFrame from KPITracker.history
#         metrics: List of metrics to animate (e.g., ['fitness', 'average_sinr'])
#         fps: Frames per second for video output
#         """
#         self.df = df
#         self.metrics = metrics
#         self.fps = fps
#         self.max_iter = df['episode'].max()
#         self.algorithms = df['algorithm'].unique()
        
#         # Style configuration
#         self.colors = plt.cm.viridis(np.linspace(0, 1, len(self.algorithms)))
#         self.markers = ['o', 's', '^', 'D', '*']  # Recycle as needed

#     def _setup_figure(self):
#         """Initialize subplots for all metrics"""
#         self.fig, self.axs = plt.subplots(
#             len(self.metrics), 1, 
#             figsize=(12, 5*len(self.metrics)))
        
#         # Initialize empty plots
#         self.lines = {}
#         for idx, metric in enumerate(self.metrics):
#             ax = self.axs[idx] if len(self.metrics) > 1 else self.axs
#             ax.set_title(metric.replace('_', ' ').title())
#             ax.set_xlabel("Iteration")
#             ax.set_ylabel(metric)
#             ax.grid(True)
            
#             # Create line for each algorithm
#             for algo, color in zip(self.algorithms, self.colors):
#                 line, = ax.plot([], [], 
#                              label=algo, 
#                              color=color,
#                              marker=self.markers[idx % len(self.markers)],
#                              markevery=5)
#                 self.lines[(algo, metric)] = line
                
#             ax.legend()

#     def _update_frame(self, frame: int):
#         """Update all plots up to current frame (iteration)"""
#         current_data = self.df[self.df['episode'] <= frame]
        
#         for (algo, metric), line in self.lines.items():
#             algo_data = current_data[
#                 (current_data['algorithm'] == algo)
#             ].sort_values('episode')
            
#             if not algo_data.empty:
#                 line.set_data(algo_data['episode'], algo_data[metric])
#                 # Auto-adjust axes
#                 self.axs[0].relim()  
#                 self.axs[0].autoscale_view()
        
#         return list(self.lines.values())

#     def animate(self):
#         """Generate and return animation object"""
#         self._setup_figure()
#         ani = animation.FuncAnimation(
#             self.fig, self._update_frame,
#             frames=range(self.max_iter + 1),
#             interval=500, # 1000//self.fps,
#             blit=True
#         )
#         return ani

#     def save_video(self, path: str = "algorithm_progression.mp4"):
#         """Render animation to video file"""
#         ani = self.animate()
#         ani.save(path, writer='ffmpeg', fps=self.fps)
#         print(f"Animation saved to {path}")

#     def show(self):
#         """Display in Jupyter notebooks"""
#         ani = self.animate()
#         return HTML(ani.to_html5_video())