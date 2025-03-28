# hybrid_trainer/live_dashboard.py
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from IPython import display  # Import inside method for Colab compatibility
import numpy as np
import time

class LiveDashboard:
    def __init__(self, network_bounds=(0, 100), algorithm_colors=None):
        self.fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "scatter3d", "rowspan": 2}, {"type": "xy"}],
                [None, {"type": "heatmap"}]
            ],
            column_widths=[0.6, 0.4],
            vertical_spacing=0.05,
            horizontal_spacing=0.05
        )
        
        self.algorithm_colors = algorithm_colors or {
            "de": "#FF6B6B",
            "aco": "#4ECDC4",
            "pso": "#45B7D1",
            "marl": "#9B59B6"
        }
        
        self._initialize_traces()
        self._add_controls()
        self._setup_layout(network_bounds)
        self.fig.update_layout(title="6G Network Optimization")
        # self.fig.show(renderer="colab")  # Force Colab rendering
        self.figure_handle = display.display(self.fig, display_id='live-dashboard')
        self.algorithm_metrics = {}  # Track metrics per algorithm
        self.fitness_traces = {}  # For fitness progression plots
        self.sinr_heatmap_trace = None  # Initialize heatmap trace reference
        
    pio.renderers.default = "colab"
    
    def _initialize_traces(self):
        """Create all visualization traces (initially hidden)"""
        
        # 3D Network Traces
        self.fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers',
            marker=dict(size=6, color='red'),
            name='Base Stations',
            visible=True
        ), row=1, col=1)
        
        self.fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers',
            marker=dict(size=3, color='blue', opacity=0.5),
            name='Users',
            visible=True
        ), row=1, col=1)

        # Metaheuristic Algorithm Traces
        self.algorithm_traces = {}
        for algo in ["de", "aco", "pso"]:
            trace = go.Scatter3d(
                x=[], y=[], z=[],
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.algorithm_colors[algo],
                    opacity=0.7
                ),
                name=f'{algo.upper()} Agents',
                visible=True
            )
            self.fig.add_trace(trace, row=1, col=1)
            self.algorithm_traces[algo] = trace

        # 2D KPI Traces
        self.fig.add_trace(go.Scatter(  # Global Reward
            x=[], y=[],
            name='Global Reward',
            line=dict(color=self.algorithm_colors["marl"]),
            visible=True
        ), row=1, col=2)
        
        self.fig.add_trace(go.Scatter(   # Fairness Index
            x=[], y=[],  
            name='Fairness Index',
            line=dict(color='#2ECC71'),
            visible=True
        ), row=1, col=2)
        
        self.fig.add_trace(go.Heatmap(
            x=[], y=[], z=[],
            colorscale='Viridis',
            name='Load Distribution',
            visible=False
        ), row=2, col=2)

    def _add_controls(self):
        """Add dropdown menus and buttons"""
        self.fig.update_layout(
            updatemenus=[
                # Main view selector
                dict(
                    buttons=list([
                        dict(
                            label=" Network View",
                            method="update",
                            args=[{"visible": [True, True] + [False]*5}]
                        ),
                        dict(
                            label=" Metaheuristic View",
                            method="update",
                            args=[{"visible": [False, False] + [True]*3 + [False]*2}]
                        ),
                        dict(
                            label=" MARL View",
                            method="update",
                            args=[{"visible": [False]*5 + [True, True, True]}]
                        )
                    ]),
                    direction="down",
                    x=0.1,
                    y=1.1,
                    showactive=True
                ),
                # Algorithm filter
                dict(
                    buttons=list([
                        dict(label="All", method="restyle", args=["visible", [True]*3]),
                        dict(label="PFO Only", method="restyle", args=["visible", [True, False, False]]),
                        dict(label="ACO Only", method="restyle", args=["visible", [False, True, False]]),
                        dict(label="PSO Only", method="restyle", args=["visible", [False, False, True]])
                    ]),
                    x=0.3,
                    y=1.1,
                    showactive=True
                )
            ],
            annotations=[
                dict(text="View Mode:", x=0.05, y=1.08, showarrow=False),
                dict(text="Algorithm Filter:", x=0.25, y=1.08, showarrow=False)
            ]
        )

    def _setup_layout(self, bounds):
        """Configure layout dimensions and labels"""
        self.fig.update_layout(
            width=1200,  # Add explicit width
            height=900,
            scene=dict(
                xaxis=dict(title='X (m)', range=bounds),
                yaxis=dict(title='Y (m)', range=bounds),
                zaxis=dict(title='Fitness', range=[0, 100]),
                aspectmode='cube'
            ),
            scene2=dict(
                xaxis=dict(title='Cell Load'),
                yaxis=dict(title='Frequency'),
                zaxis=dict(title='Users Served')
            ),
            margin=dict(l=50, r=50, b=50, t=50),
            legend=dict(x=1.1, y=1.0)
        )
        
        self.fig.update_xaxes(title_text="Iteration", row=1, col=2)
        self.fig.update_yaxes(title_text="Reward", row=1, col=2)
        self.fig.update_xaxes(title_text="Iteration", row=2, col=2)
        self.fig.update_yaxes(title_text="Fairness", row=2, col=2)

    def update(self, env_state: dict, metrics: dict, phase: str = "metaheuristic"):
        """Universal update method for all visualization components"""
        # Add debug prints in LiveDashboard.update()
        print("Updating dashboard with:", len(env_state["metaheuristic_agents"]), "agents")
        print("Current metrics keys:", metrics.keys())
        # Update network view (BS and UE positions)
        self.update_network_state(env_state["base_stations"],env_state["users"])
        
        if phase == "metaheuristic":
            # Update algorithm agents and metrics
            self.update_metaheuristic(
                algorithm=metrics.get("algorithm", "de"),
                positions=metrics.get("positions", []),
                fitness=metrics.get("fitness", [])
            )
        elif phase == "marl":
            # Update MARL-specific visualizations
            self.update_marl(
                associations=env_state["users"],
                rewards=np.array(metrics.get("rewards", [])).astype(np.float32),
                fairness=float(metrics.get("fairness", 0.0))
                )
        
        # Optional: Force refresh in Colab
        # self.fig.show(renderer="colab")
        # Colab-optimized rendering
        display.clear_output(wait=True)
        display.display(self.fig)
        
        # Add small delay to prevent DOM overflow        
        time.sleep(0.3)  # 300ms interval between updates

    
    def update_network_state(self, base_stations, users):
        """Update 3D network visualization"""
        
        with self.fig.batch_update():
            # Base stations (load as z-axis)
            self.fig.data[0].x = [bs['position'][0] for bs in base_stations]
            self.fig.data[0].y = [bs['position'][1] for bs in base_stations]
            self.fig.data[0].z = [bs['load'] for bs in base_stations]
            self.fig.data[0].marker.size = [bs["load"]*5 for bs in base_stations]  # Scale for visibility
            # Users
            self.fig.data[1].x = [u['position'][0] for u in users]
            self.fig.data[1].y = [u['position'][1] for u in users]
            self.fig.data[1].z = [u.get('sinr', 0) for u in users]
            
        print(f"Updating network with {len(base_stations)} BS, {len(users)} UEs")
        print("Example BS data:", base_stations[0])
        print("Example UE data:", users[0])    
        self.fig.show(renderer="colab")  # Refresh display
        
    def update_metaheuristic(self, algorithm, positions, fitness):
        """Update metaheuristic agents visualization"""
        trace = self.algorithm_traces[algorithm]
        with self.fig.batch_update():
            trace.x = positions[:, 0]
            trace.y = positions[:, 1]
            trace.z = fitness
            trace.marker.color = fitness
            trace.marker.colorscale = 'Viridis'

    def update_marl(self, associations, rewards, fairness):
        """Update MARL-related visualizations"""
        with self.fig.batch_update():
            # Heatmap (convert associations to 2D grid)
            x_bins = np.linspace(0, 100, 20)
            y_bins = np.linspace(0, 100, 20)
            heatmap, _, _ = np.histogram2d(
                [u['position'][0] for u in associations],
                [u['position'][1] for u in associations],
                bins=[x_bins, y_bins]
            )
            self.fig.data[5].x = x_bins
            self.fig.data[5].y = y_bins
            self.fig.data[5].z = heatmap
            
            # KPI curves
            self.fig.data[3].x = list(range(len(rewards)))
            self.fig.data[3].y = rewards
            self.fig.data[4].x = list(range(len(fairness)))
            self.fig.data[4].y = fairness

    def show(self):
        """Display the dashboard"""
        self.fig.show()

    def save(self, filename="dashboard.html"):
        """Save as standalone HTML file"""
        self.fig.write_html(filename)
    
    def update_algorithm_metrics(self, algorithm: str, metrics: dict):
        """Update visualization with algorithm-specific metrics"""
        self.algorithm_metrics[algorithm] = metrics
        
        # Example: Update fitness plot
        if "fitness" in metrics:
            self._update_fitness_plot(algorithm, metrics["fitness"])
        
        # Example: Update SINR heatmap
        if "average_sinr" in metrics:
            self._update_sinr_heatmap(algorithm, metrics["average_sinr"])
        else:
            print(f"Warning: No SINR data for {algorithm}")
            
    def _update_fitness_plot(self, algorithm: str, fitness: float):
        """Example: Append fitness to a line chart"""
        if algorithm not in self.fitness_traces:
            self.fitness_traces[algorithm] = go.Scatter(
                x=[], y=[], name=f"{algorithm.upper()} Fitness"
            )
            self.fig.add_trace(self.fitness_traces[algorithm], row=1, col=2)
        
        trace = self.fitness_traces[algorithm]
        trace.x = list(trace.x) + [len(trace.x)]
        trace.y = list(trace.y) + [fitness]
    
    def _update_sinr_heatmap(self, algorithm: str, sinr_value: float):
        """Update SINR heatmap visualization (simplified example)"""
        if self.sinr_heatmap_trace is None:
            # Initialize heatmap trace on first call
            self.sinr_heatmap_trace = go.Heatmap(
                z=[[sinr_value]],  # Start with initial value
                colorscale='Viridis',
                name=f'{algorithm.upper()} SINR'
            )
            self.fig.add_trace(self.sinr_heatmap_trace, row=1, col=2)
        else:
            # Update existing trace (append new value)
            self.sinr_heatmap_trace.z = np.vstack([
                self.sinr_heatmap_trace.z,
                [sinr_value]
            ])
    
    def finalize_visualizations(self):
        """Save final plots and clean up resources"""
        self.fig.write_html("results/final_dashboard.html")
        # plt.close('all')  # Close matplotlib figures if used