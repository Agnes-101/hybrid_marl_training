# hybrid_trainer/live_dashboard.py
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
from IPython import display

class LiveDashboard:
    def __init__(self, network_bounds=(0, 100), algorithm_colors=None):
        # Initialize figure with subplot grid
        # During initialization
        self.x_grid = np.linspace(0, 100, 100)  # X-axis coordinates
        self.y_grid = np.linspace(0, 100, 100)  # Y-axis coordinates
        self.xx, self.yy = np.meshgrid(self.x_grid, self.y_grid)
        
        self.algorithm_colors = algorithm_colors or {
            "pfo": "#A020F0",
            "de": "#FF6B6B",
            "aco": "#4ECDC4",
            "pso": "#45B7D1",
            "marl": "#9B59B6"
        }
        self.fig = sp.make_subplots(
            rows=5, cols=2,
            row_heights=[0.05, 0.05, 0.5, 0.5, 0.25],  # Prioritize first and last rows
            specs=[
                [{"type": "scattergl", "rowspan": 3}, {"type": "indicator"}],
                [None, {"type": "indicator"}],
                [None, {"type": "bar"}],
                [{"type": "heatmap", "rowspan": 2}, {"type": "scatter", "rowspan": 2}],
                [None, None] 
            ],
            column_widths=[0.7, 0.3],
            vertical_spacing= 0.1 #0.05
        )
        
        # Initialize all traces
        self._init_traces(network_bounds)
        self._add_controls()
        
        # Display initial figure
        self.figure_widget = go.FigureWidget(self.fig)
        #  display.display(self.figure_widget)
        
        # Track UI state
        self.current_view = "network"  # Initialize with default view
        self.overlays = {"Overlay": False, "associations": False}  # Track overlay states
        
        # Track active button indices for each updatemenu
        self.view_menu_active = 0  # Default: "Network" view
        self.overlay_menu_active = None  # No overlay active by default


    def _init_traces(self, bounds):
        """Initialize all visualization traces"""
        # Main View (Column 1)
        # Network
        # Trace 0: Base Stations
        self.fig.add_trace(go.Scattergl(
            x=[], y=[], mode='markers', visible=True, name='Base Stations',
            marker=dict(symbol='square', size=15, color='#BD0D0D', opacity=1.0)),
            row=1, col=1
        )
        
        # Trace 1: Users
        self.fig.add_trace(go.Scattergl(
            x=[], y=[], mode='markers',visible= True, name='Users',
            marker=dict(size=6, color='blue', opacity=0.4)),
            row=1, col=1
        )
        
        # Trace 2: Metaheuristic
        for algo in ["de", "pso", "aco"]:
            self.fig.add_trace(go.Scattergl(
                x=[], y=[],  mode='markers', visible=False, name=f'{algo.upper()} Agents',
                marker=dict(size=8)), row=1, col=1
            )
        
        # Trace 3: MARL Associations
        self.fig.add_trace(go.Scattergl(
            x=[], y=[], mode='lines', visible=False,
            line=dict(width=1), name='Associations'), row=1, col=1
        )

        # Network KPIs (Column 2)
        # Trace 4: Connected Users
        self.fig.add_trace(go.Indicator(
            mode="number+delta", name= 'Connected Users', visible=True,title="Connected Users",
            number=dict(font=dict(size=40))), row=1, col=2)
        
        # Trace 5 : Average SINR
        self.fig.add_trace(go.Indicator(
            mode="gauge", name='Avg SINR', visible=True,title="Avg SINR",
            gauge=dict(axis=dict(range=[0, 30], tickfont_size=10),  # Smaller ticks
        bar_thickness=0.3)), row=2, col=2)
        
        # Trace 6: BS load Bar
        self.fig.add_trace(go.Bar(
            x=[], y=[], name='BS Load', visible=True), row=3, col=2)

        # Phase KPIs (Row 4)
        # Trace 7: SINR Heatmap
        self.fig.add_trace(go.Heatmap(
            x=[], y=[], name= 'SINR Heatmap', colorscale='Viridis', showscale=False,
            visible=False), row=4, col=1)
        # Trace 8: Fitness
        self.fig.add_trace(go.Scatter(
            x=[], y=[], name='Fitness', visible=False), row=4, col=2)
        # Trace 9: Reward
        self.fig.add_trace(go.Scatter(
            x=[], y=[], name='Reward', visible=False), row=4, col=2)
        
        # After adding traces, update axes:
        self.fig.update_xaxes(title_text="X Coordinate (meters)", row=1, col=1)
        self.fig.update_yaxes(title_text="Y Coordinate (meters)", row=1, col=1) 
        self.fig.update_xaxes(title_text="X Position (meters)", row=4, col=1)
        self.fig.update_yaxes(title_text="Y Position (meters)", row=4, col=1)  
        self.fig.update_xaxes(title_text="Base Station ID", row=3, col=2)
        self.fig.update_yaxes(title_text="Load (%)", row=3, col=2)
        
        
    
    def _add_controls(self):
        """Add interactive controls"""
        self.fig.update_layout(
            height=900,  # Increase total height
            width=1400,
            margin=dict(t=150),  # Add top margin for buttons
            updatemenus=[
                dict(
                    buttons=[
                        dict(label="Network", method="update",
                            args=[{"visible": [True, True]+[False]*4+[True]*6},
                                {"title": "Network View"}]),
                        dict(label="Metaheuristic", method="update",
                            args=[{"visible": [False]*2+[True]*3+[False]*2},
                                {"title": "Metaheuristic View"}]),
                        dict(label="MARL", method="update",
                            args=[{"visible": [False]*5+[True]+[False]},
                                {"title": "MARL View"}])
                    ],
                    active=0,  # "Network" view is default
                showactive=True,  # Highlight active button
                direction="down", x=0.05, y=1.05, xanchor="left", yanchor="bottom" 
                ),
                dict(
                    buttons=[
                        dict(label="Overlay", method="restyle",
                            args=[{"visible": [True]*8 + [True, True]}]),
                        dict(label="Associations", method="restyle",
                            args=[{"visible": [True]*8 + [True, True]}])
                    ],
                    active=None,  # No overlay active by default
                showactive=True, x=0.35, y=1.05, xanchor="left", yanchor="bottom"
                )
            ]
        )
        
    def _get_trace_by_name(self, name: str):
        """Get trace by name (safer than indices)"""
        for trace in self.fig.data:
            if trace.name == name:
                return trace
        raise ValueError(f"Trace '{name}' not found")
    
    def _save_button_states(self):
        """Save current active button indices for all menus"""
        self.view_menu_active = self.fig.layout.updatemenus[0].active
        self.overlay_menu_active = self.fig.layout.updatemenus[1].active

    def _restore_button_states(self):
        """Reapply saved button states to retain UI settings"""
        with self.fig.batch_update():
            self.fig.layout.updatemenus[0].active = self.view_menu_active
            self.fig.layout.updatemenus[1].active = self.overlay_menu_active
    
    def update(self, phase: str, data: dict): 
        # Save current button states
        self._save_button_states()
        
        """Main update entry point"""
        with self.fig.batch_update():
            # Update main view
            # Extract env_state and metrics from data
            env_state = data.get("env_state", {})
            metrics = data.get("metrics", {})
            
            self._update_network(env_state)
            self._update_network_kpis(env_state)
            
            if phase != self.current_view:
                self._handle_view_change(phase)
            
            if phase == "metaheuristic":
                self._update_metaheuristic(metrics)
            elif phase == "marl":
                self._update_marl(env_state)
            
            # Update persistent KPIs
            self._update_phase_kpis(phase, metrics)
            # Update current state tracking
            self.current_view = phase
            
        # Restore button states to previous UI configuration
        self._restore_button_states()
    
    def _update_network(self, env_state):
        # """Update base stations and users"""
        # # Base Stations
        # self.fig.data[0].x = [bs["position"][0] for bs in env_state["base_stations"]]
        # self.fig.data[0].y = [bs["position"][1] for bs in env_state["base_stations"]]
        # self.fig.data[0].marker.size = [float(bs["load"]) * 10 for bs in env_state["base_stations"]]
        
        bs_trace = self.fig.data[0]
        bs_trace.x = [bs["position"][0] for bs in env_state["base_stations"]]
        bs_trace.y = [bs["position"][1] for bs in env_state["base_stations"]]
        bs_trace.marker.size = [float(bs["load"]) * 10 for bs in env_state["base_stations"]]

        
        # Users
        self.fig.data[1].x = [ue["position"][0] for ue in env_state["users"]]
        self.fig.data[1].y = [ue["position"][1] for ue in env_state["users"]]
        # self.fig.data[1].marker.color = [ue["sinr"] for ue in env_state["users"]]
        self.fig.data[1].marker.color = [float(ue["sinr"]) for ue in env_state["users"]]  # Force float
        
    
    def _update_metaheuristic(self, metrics):
        """Update metaheuristic agents"""
        algo_idx = {"de": 2, "pso": 3, "aco": 4}[metrics['algorithm']]
        
        # Extract x and y from the list of positions
        x = [pos[0] for pos in metrics['positions']]  # First element of each position
        y = [pos[1] for pos in metrics['positions']]  # Second element of each position
        
        self.fig.data[algo_idx].x = x
        self.fig.data[algo_idx].y = y
        self.fig.data[algo_idx].marker.color = metrics['fitness']
        self.fig.data[algo_idx].marker.colorscale = 'Viridis'

    def _update_marl(self, env_state):
        """Update MARL associations"""
        x, y = [], []
        for ue, bs in env_state["associations"].items():
            ue_pos = next(u["position"] for u in env_state["users"] if u["id"] == ue)
            bs_pos = next(b["position"] for b in env_state["base_stations"] if b["id"] == bs)
            x.extend([ue_pos[0], bs_pos[0], None])
            y.extend([ue_pos[1], bs_pos[1], None])
        
        self.fig.data[5].x = x
        self.fig.data[5].y = y
        self.fig.data[5].line.color = [ue["sinr"] for ue in env_state["users"]]

    # def _update_network_kpis(self, metrics):
    #     """Update persistent network metrics"""
    #     self.fig.data[6].value = metrics.get('connected_users', 0)
    #     self.fig.data[7].value = metrics.get('avg_sinr', 0)
    #     self.fig.data[8].x = [bs['id'] for bs in metrics['base_stations']]
    #     self.fig.data[8].y = [bs['load'] for bs in metrics['base_stations']]
        
    def _update_network_kpis(self, env_state: dict):
        
        """Update network KPIs using env_state"""
        # Get traces by name
        connected_users_trace = self._get_trace_by_name('Connected Users')
        avg_sinr_trace = self._get_trace_by_name('Avg SINR')
        bs_load_trace = self._get_trace_by_name('BS Load')
        sinr_heatmap = self._get_trace_by_name('SINR Heatmap')

        # Calculate and update connected users
        connected_users = sum(1 for ue in env_state["users"] if ue["associated_bs"] is not None)
        print(f"Connected Users: {connected_users}")  # Should be >0

        connected_users_trace.value = connected_users

        # Calculate and update average SINR
        sinr_values = [ue["sinr"] for ue in env_state["users"]]
        avg_sinr = np.mean(sinr_values) if sinr_values else 0
        print(f"Avg SINR: {avg_sinr}")  # Should be a float
        avg_sinr_trace.value = avg_sinr
        avg_sinr_trace.visible=True

        # Update BS Load bar chart
        bs_load_trace.x = [bs["id"] for bs in env_state["base_stations"]]
        bs_load_trace.y = [bs["load"] for bs in env_state["base_stations"]]
        bs_loads = [bs["load"] for bs in env_state["base_stations"]]
        print(f"BS Loads: {bs_loads}")  # Should be a list of floats
        bs_load_trace.visible=True
        
        # Calculate grid SINR
        grid_sinr = self.calculate_grid_sinr(env_state)
        
        # Update heatmap trace
        sinr_heatmap.x = self.x_grid  # Grid X-coordinates   self.fig.data[2]
        sinr_heatmap.y = self.y_grid  # Grid Y-coordinates
        sinr_heatmap.z = grid_sinr    # SINR values across grid
        sinr_heatmap.visible = True
        
    def _apply_view(self, view_name: str):
        """Restore the specified view configuration"""
        trace_visibility = {
            "network": ["Base Stations", "Users"],
            "metaheuristic": ["DE Agents", "PSO Agents", "ACO Agents"],
            "marl": ["Associations"]
        }
        
        # Set visibility for all traces
        for trace in self.fig.data:
            trace.visible = trace.name in trace_visibility.get(view_name, [])
            
    def _apply_overlays(self, overlays: dict):
        """Restore overlay states"""
        # SINR Heatmap
        sinr_trace = next(t for t in self.fig.data if t.name == "SINR Heatmap")
        sinr_trace.visible = overlays["sinr"]
        
        # Associations
        assoc_trace = next(t for t in self.fig.data if t.name == "Associations")
        assoc_trace.visible = overlays["associations"]


    # def _update_phase_kpis(self, phase, metrics):
    #     """Update phase-specific KPIs"""
    #     if phase == "metaheuristic":
    #         self.fig.data[9].visible = True
    #         self.fig.data[10].visible = False
    #         self.fig.data[9].x = list(range(len(metrics['fitness'])))
    #         self.fig.data[9].y = metrics['fitness']
    #     elif phase == "marl":
    #         self.fig.data[9].visible = False
    #         self.fig.data[10].visible = True
    #         self.fig.data[10].x = list(range(len(metrics['reward'])))
    #         self.fig.data[10].y = metrics['reward']
          
    def _update_phase_kpis(self, phase:str, metrics):
        """Handle phase-specific KPI updates"""
        # Clear previous phase traces
        self.figure_widget.data[9].visible = False  # Fitness plot
        self.figure_widget.data[10].visible = False  # Reward plot
        
        if phase == "metaheuristic":
            # Update and show metaheuristic KPIs
            self._update_fitness_plot(metrics)
            # self._update_diversity_heatmap(metrics)
            
        elif phase == "marl":
            # Update and show MARL KPIs
            self._update_reward_plot(metrics)
            self._update_entropy_plot(metrics)

    def _update_fitness_plot(self, metrics):
        """Update fitness progression plot"""
        fitness = metrics.get('fitness_history', [])
        self.figure_widget.data[9].x = list(range(len(fitness)))
        self.figure_widget.data[9].y = fitness
        self.figure_widget.data[9].visible = True

    def _update_reward_plot(self, metrics):
        """Update MARL reward plot"""
        rewards = metrics.get('episode_rewards', [])
        self.figure_widget.data[10].x = list(range(len(rewards)))
        self.figure_widget.data[10].y = rewards
        self.figure_widget.data[10].visible = True
        
    def calculate_grid_sinr(self, env_state):
        """Estimate SINR across the grid"""
        grid_sinr = np.zeros_like(self.xx)
        for bs in env_state["base_stations"]:
            # Calculate signal strength from each BS to grid points
            bs_x, bs_y = bs["position"]
            distance = np.sqrt((self.xx - bs_x)**2 + (self.yy - bs_y)**2)
            signal_strength = 1 / (1 + distance**2)  # Simplified path loss model
            grid_sinr += signal_strength
        return grid_sinr
    
    def _update_diversity_heatmap(self, metrics):
        """Update population diversity visualization"""
        positions = metrics.get('agent_positions', [])
        if positions:
            x = [p[0] for p in positions]
            y = [p[1] for p in positions]
            self.figure_widget.data[11].x = x
            self.figure_widget.data[11].y = y
            self.figure_widget.data[11].visible = True

    def _update_entropy_plot(self, metrics):
        """Update policy entropy visualization"""
        entropy = metrics.get('policy_entropy', [])
        self.figure_widget.data[12].x = list(range(len(entropy)))
        self.figure_widget.data[12].y = entropy
        self.figure_widget.data[12].visible = True
        
    def _handle_view_change(self, new_view):
        """Handle visibility changes between views"""
        # Hide all traces first
        for trace in self.fig.data:
            trace.visible = False
            
        # Hide previous view
        if self.current_view == "metaheuristic":
            for i in [2,3,4]: self.fig.data[i].visible = False
        elif self.current_view == "marl":
            self.fig.data[5].visible = False
        
        # Show new view
        self.current_view = new_view
    # def _handle_view_change(self, new_view: str):
    #     """Properly toggle visibility for views"""
    #     # Hide all non-essential traces
    #     for trace in self.fig.data:
    #         if trace.name not in ['Base Stations', 'Users']:
    #             trace.visible = False
        
    #     # Show traces for the new view
    #     if new_view == "metaheuristic":
    #         for algo in ["de", "pso", "aco"]:
    #             self._get_trace_by_name(f"{algo.upper()} Agents").visible = True
    #     elif new_view == "marl":
    #         self._get_trace_by_name("Associations").visible = True
        
    #     self.current_view = new_view
        
    def save(self, filename="results/final_dashboard.html"):
        """Save dashboard to HTML file"""
        self.fig.write_html(filename)
        print(f"Dashboard saved to {filename}")
        
# # hybrid_trainer/live_dashboard.py
# import plotly.graph_objects as go
# import plotly.io as pio
# from plotly.subplots import make_subplots
# from IPython import display  # Import inside method for Colab compatibility
# import numpy as np
# import time

# class LiveDashboard:
#     def __init__(self, network_bounds=(0, 100), algorithm_colors=None):
#         self.fig = make_subplots(
#             rows=2, cols=2,
#             specs=[
#                 [{"type": "scatter3d", "rowspan": 2}, {"type": "xy"}],
#                 [None, {"type": "heatmap"}]
#             ],
#             column_widths=[0.6, 0.4],
#             vertical_spacing=0.05,
#             horizontal_spacing=0.05
#         )
        
#         self.algorithm_colors = algorithm_colors or {
#             "pfo": "#A020F0",
#             "de": "#FF6B6B",
#             "aco": "#4ECDC4",
#             "pso": "#45B7D1",
#             "marl": "#9B59B6"
            
#         }
#         # print("Algorithm Colors:", self.algorithm_colors)

#         self._initialize_traces()
#         self.env_version = -1  # Initialize to -1 (no state yet)
#         self._add_controls()
#         self._setup_layout(network_bounds)
#         self.env_version = 0  # Track state changes
#         self.fig.update_layout(title="6G Network Optimization")
#         # self.fig.show(renderer="colab")  # Force Colab rendering
#         self.figure_handle = display.display(self.fig, display_id='live-dashboard')
#         self.algorithm_metrics = {}  # Track metrics per algorithm
#         self.fitness_traces = {}  # For fitness progression plots
#         self.sinr_heatmap_trace = None  # Initialize heatmap trace reference
        
#     pio.renderers.default = "colab"
    
#     def _initialize_traces(self):
#         """Create all visualization traces (initially hidden)"""
        
#         # 3D Network Traces
#         self.fig.add_trace(go.Scatter3d(
#             x=[], y=[] , z=[],
#             mode='markers',
#             marker=dict(size=6, color='red'),
#             name='Base Stations',
#             visible=True
#         ), row=1, col=1)
        
#         self.fig.add_trace(go.Scatter3d(
#             x=[], y=[], z=[],
#             mode='markers',
#             marker=dict(size=3, color='blue', opacity=0.5),
#             name='True',
#             visible=False
#         ), row=1, col=1)

#         # Metaheuristic Algorithm Traces
#         self.algorithm_traces = {}
#         for algo in ["pfo", "aco", "pso", "de"]:
#             trace = go.Scatter3d(
#                 x=[], y=[], z=[],
#                 mode='markers',
#                 marker=dict(
#                     size=8,
#                     color=self.algorithm_colors[algo],
#                     opacity=0.7
#                 ),
#                 name=f'{algo.upper()} Agents',
#                 visible=False
#             )
#             self.fig.add_trace(trace, row=1, col=1)
#             self.algorithm_traces[algo] = trace

#         # 2D KPI Traces
#         self.fig.add_trace(go.Scatter(  # Global Reward
#             x=[], y=[],
#             name='Global Reward',
#             line=dict(color="#9B59B6"), #,self.algorithm_colors["marl"]
#             visible=False
#         ), row=1, col=2)
        
#         self.fig.add_trace(go.Scatter(   # Fairness Index
#             x=[], y=[],  
#             name='Fairness Index',
#             line=dict(color='#2ECC71'),
#             visible=False
#         ), row=1, col=2)
        
#         self.fig.add_trace(go.Heatmap(
#             x=[], y=[], z=[],
#             colorscale='Viridis',
#             name='Load Distribution',
#             visible=False
#         ), row=2, col=2)

#     def _add_controls(self):
#         """Add dropdown menus and buttons"""
#         self.fig.update_layout(
#             updatemenus=[
#                 # Main view selector
#                 dict(
#                     buttons=list([
#                         dict(
#                             label=" Network View",
#                             method="update",
#                             args=[{"visible": [True, True] + [False]*7}]
#                         ),
#                         dict(
#                             label=" Metaheuristic View",
#                             method="update",
#                             args=[{"visible": [False, False] + [True]*4 + [False]*3}]
#                         ),
#                         dict(
#                             label=" MARL View",
#                             method="update",
#                             args=[{"visible": [False]*6 + [True, True, True]}]
#                         )
#                     ]),
#                     direction="down",
#                     x=0.1,
#                     y=1.1,
#                     showactive=True
#                 ),
#                 # Algorithm filter
#                 dict(
#                     buttons=list([
#                         dict(label="All", method="restyle", args=["visible", [True]*3]),
#                         dict(label="DE Only", method="restyle", args=["visible", [True, False, False]]),
#                         dict(label="ACO Only", method="restyle", args=["visible", [False, True, False]]),
#                         dict(label="PSO Only", method="restyle", args=["visible", [False, False, True]])
#                     ]),
#                     x=0.3,
#                     y=1.1,
#                     showactive=True
#                 )
#             ],
#             annotations=[
#                 dict(text="View Mode:", x=0.05, y=1.08, showarrow=False),
#                 dict(text="Algorithm Filter:", x=0.25, y=1.08, showarrow=False)
#             ]
#         )

#     def _setup_layout(self, bounds):
#         """Configure layout dimensions and labels"""
#         self.fig.update_layout(
#             width=1200,  # Add explicit width
#             height=900,
#             scene=dict(
#                 xaxis=dict(title='X (m)', range=bounds),
#                 yaxis=dict(title='Y (m)', range=bounds),
#                 zaxis=dict(title='Fitness', range=[0, 100]),
#                 aspectmode='cube'
#             ),
#             scene2=dict(
#                 xaxis=dict(title='Cell Load'),
#                 yaxis=dict(title='Frequency'),
#                 zaxis=dict(title='Users Served')
#             ),
#             margin=dict(l=50, r=50, b=50, t=50),
#             legend=dict(x=1.1, y=1.0)
#         )
        
#         self.fig.update_xaxes(title_text="Iteration", row=1, col=2)
#         self.fig.update_yaxes(title_text="Reward", row=1, col=2)
#         self.fig.update_xaxes(title_text="Iteration", row=2, col=2)
#         self.fig.update_yaxes(title_text="Fairness", row=2, col=2)

#     def update(self, env_state: dict, metrics: dict, phase: str = "metaheuristic"):
#         from IPython import display
#         import time
        
#         # Check version before updating
#         if env_state["version"] > self.env_version:
#             self._update_network_state(env_state)
#             self.env_version = env_state["version"]  # Update to latest
            
#         print(f"Dashboard version: {self.env_version}, Env version: {env_state['version']}")
        
#         # Clear previous output
#         display.clear_output(wait=True)
    
#         """Universal update method for all visualization components"""
#         # Add debug prints in LiveDashboard.update()
#         print("Updating dashboard with:", len(env_state["metaheuristic_agents"]), "agents")
#         print("Current metrics keys:", metrics.keys())
#         # Update network view (BS and UE positions)
#         self.update_network_state(env_state["base_stations"],env_state["users"])
        
#         if phase == "metaheuristic":
#             # Update algorithm agents and metrics
#             self.update_metaheuristic(
#                 algorithm=metrics.get("algorithm", "de"),
#                 positions=metrics.get("positions", []),
#                 fitness=metrics.get("fitness", [])
#             )
#         elif phase == "marl":
#             # Update MARL-specific visualizations
#             self.update_marl(
#                 associations=env_state["users"],
#                 rewards=np.array(metrics.get("rewards", [])).astype(np.float32),
#                 fairness=float(metrics.get("fairness", 0.0))
#                 )
        
#         # Optional: Force refresh in Colab
#         # self.fig.show(renderer="colab")
#         # Colab-optimized rendering
#         # Clear previous output
#         # display.clear_output(wait=True)
#         display.display(self.fig)
        
#         # Add small delay to prevent DOM overflow        
#         time.sleep(0.3)  # 300ms interval between updates

    
#     def update_network_state(self, base_stations, users):
#         """Update 3D network visualization"""
        
#         with self.fig.batch_update():
#             # Base stations (load as z-axis)
#             self.fig.data[0].x = [bs['position'][0] for bs in base_stations]
#             self.fig.data[0].y = [bs['position'][1] for bs in base_stations]
#             self.fig.data[0].z = [bs['load'] for bs in base_stations]
#             self.fig.data[0].marker.size = [bs["load"]*5 for bs in base_stations]  # Scale for visibility
#             # Users
#             self.fig.data[1].x = [u['position'][0] for u in users]
#             self.fig.data[1].y = [u['position'][1] for u in users]
#             self.fig.data[1].z = [u.get('sinr', 0) for u in users]
            
#         print(f"Updating network with {len(base_stations)} BS, {len(users)} UEs")
#         self.fig.show(renderer="colab")  # Refresh display
        
#     def update_metaheuristic(self, algorithm, positions, fitness):
#         """Update metaheuristic agents visualization"""
#         positions = np.array(positions)  # Convert list to array
#         fitness = np.array(fitness)
    
#         trace = self.algorithm_traces[algorithm]
#         with self.fig.batch_update():
#             trace.x = positions[:, 0]
#             trace.y = positions[:, 1]
#             trace.z = fitness
#             trace.marker.color = fitness
#             trace.marker.colorscale = 'Blues' # 'Viridis'
            
        
#     def update_marl(self, associations, rewards, fairness):
#         """Update MARL-related visualizations"""
#         with self.fig.batch_update():
#             # Heatmap (convert associations to 2D grid)
#             x_bins = np.linspace(0, 100, 20)
#             y_bins = np.linspace(0, 100, 20)
#             heatmap, _, _ = np.histogram2d(
#                 [u['position'][0] for u in associations],
#                 [u['position'][1] for u in associations],
#                 bins=[x_bins, y_bins]
#             )
#             self.fig.data[5].x = x_bins
#             self.fig.data[5].y = y_bins
#             self.fig.data[5].z = heatmap
            
#             # KPI curves
#             self.fig.data[3].x = list(range(len(rewards)))
#             self.fig.data[3].y = rewards
#             self.fig.data[4].x = list(range(len(fairness)))
#             self.fig.data[4].y = fairness

#     def show(self):
#         """Display the dashboard"""
#         self.fig.show()

#     def save(self, filename="dashboard.html"):
#         """Save as standalone HTML file"""
#         self.fig.write_html(filename)
    
#     def update_algorithm_metrics(self, algorithm: str, metrics: dict):
#         """Update visualization with algorithm-specific metrics"""
#         self.algorithm_metrics[algorithm] = metrics
        
#         # Example: Update fitness plot
#         if "fitness" in metrics:
#             self._update_fitness_plot(algorithm, metrics["fitness"])
        
#         # Example: Update SINR heatmap
#         if "average_sinr" in metrics:
#             self._update_sinr_heatmap(algorithm, metrics["average_sinr"])
#         else:
#             print(f"Warning: No SINR data for {algorithm}")
            
#     def _update_fitness_plot(self, algorithm: str, fitness: float):
#         """Example: Append fitness to a line chart"""
#         if algorithm not in self.fitness_traces:
#             self.fitness_traces[algorithm] = go.Scatter(
#                 x=[], y=[], name=f"{algorithm.upper()} Fitness"
#             )
#             self.fig.add_trace(self.fitness_traces[algorithm], row=1, col=2)
        
#         trace = self.fitness_traces[algorithm]
#         trace.x = list(trace.x) + [len(trace.x)]
#         trace.y = list(trace.y) + [fitness]
    
#     def _update_sinr_heatmap(self, algorithm: str, sinr_value: float):
#         """Update SINR heatmap visualization (simplified example)"""
#         """Update SINR heatmap with validation"""
#         if np.isnan(sinr_value):
#             sinr_value = 0  # Or other default
            
#         if self.sinr_heatmap_trace is None:
#             # Initialize heatmap trace on first call
#             self.sinr_heatmap_trace = go.Heatmap(
#                 z=[[sinr_value]],  # Start with initial value
#                 colorscale='Viridis',
#                 name=f'{algorithm.upper()} SINR'
#             )
#             self.fig.add_trace(self.sinr_heatmap_trace, row=1, col=2)
#         else:
#             # Update existing trace (append new value)
#             new_z = np.vstack([self.sinr_heatmap_trace.z, [sinr_value]])
#             self.sinr_heatmap_trace.z = new_z[-20:]  # Keep last 20 values
    
#     def finalize_visualizations(self):
#         """Save final plots and clean up resources"""
#         self.fig.write_html("results/final_dashboard.html")
       