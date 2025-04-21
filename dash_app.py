# streamlit_app.py (modified)
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np

# Path configuration
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

# dash.py
import dash
from dash import dcc, html, Input, Output, State, callback
import pandas as pd
import plotly.express as px
from dash.exceptions import PreventUpdate
import time
import numpy as np

# Import your custom modules
from core.envs.custom_channel_env import NetworkEnvironment
from core.hybrid_trainer.metaheuristic_opt import run_metaheuristic
from core.hybrid_trainer.kpi_logger import WebKPILogger

app = dash.Dash(__name__)

# Custom CSS styles
styles = {
    'header': {'textAlign': 'center', 'padding': '20px'},
    'controls': {'width': '25%', 'float': 'left', 'padding': '20px'},
    'main': {'width': '75%', 'float': 'left', 'padding': '20px'},
    'metric-card': {'border': '1px solid #ddd', 'padding': '10px', 'margin': '10px', 'borderRadius': '5px'},
    'chart': {'height': '300px'}
}

app.layout = html.Div([
    dcc.Store(id='session-data', data={'running': False}),
    dcc.Interval(id='update-interval', interval=1000, disabled=True),
    
    html.H1("6G MARL Optimization Dashboard", style=styles['header']),
    
    html.Div([
        html.Div([
            html.H3("Configuration"),
            html.Label("Algorithm"),
            dcc.Dropdown(
                id='algorithm',
                options=[
                    {'label': 'Polar Fox', 'value': 'pfo'},
                    {'label': 'Particle Swarm', 'value': 'pso'},
                    {'label': 'Ant Colony', 'value': 'aco'}
                ],
                value='pfo'
            ),
            
            html.Br(),
            html.Label("Number of Base Stations"),
            dcc.Slider(id='num-bs', min=5, max=50, step=5, value=10),
            
            html.Br(),
            html.Label("Number of UEs"),
            dcc.Slider(id='num-ue', min=20, max=200, step=10, value=50),
            
            html.Br(),
            html.Button("Run Optimization", id='run-button')
        ], style=styles['controls']),
        
        html.Div([
            html.Div(id='kpi-cards', style={'display': 'flex'}),
            html.Div([
                dcc.Graph(id='fitness-chart', style=styles['chart']),
                dcc.Graph(id='sinr-chart', style=styles['chart']),
                dcc.Graph(id='fairness-chart', style=styles['chart'])
            ], style={'display': 'flex'}),
            dcc.Graph(id='network-map', style={'height': '500px'})
        ], style=styles['main'])
    ])
])

# Same callbacks as previous version (modified for style changes)
@callback(
    [Output('update-interval', 'disabled'),
     Output('session-data', 'data'),
     Output('run-button', 'disabled')],
    [Input('run-button', 'n_clicks')],
    [State('session-data', 'data'),
     State('algorithm', 'value'),
     State('num-bs', 'value'),
     State('num-ue', 'value')],
    prevent_initial_call=True
)
def start_optimization(n_clicks, data, algorithm, num_bs, num_ue):
    if not data['running']:
        env_config = {
                        "num_ue": num_ue,
                        "num_bs": num_bs
                    }
        data.update({
            'running': True,
            'progress': {'iteration': 0, 'metrics': {}, 'positions': []},
            'result': None,
            'env': NetworkEnvironment(config=env_config, log_kpis=False).__dict__,
            'kpi_logger': WebKPILogger().__dict__,
            'start_time': time.time()
        })
        return False, data, True
    return dash.no_update

@callback(
    [Output('kpi-cards', 'children'),
     Output('fitness-chart', 'figure'),
     Output('sinr-chart', 'figure'),
     Output('fairness-chart', 'figure'),
     Output('network-map', 'figure'),
     Output('session-data', 'data', allow_duplicate=True)],
    [Input('update-interval', 'n_intervals')],
    [State('session-data', 'data')],
    prevent_initial_call=True
)
def update_ui(n, data):
    if not data['running']:
        raise PreventUpdate
    
    # Simulate optimization progress (replace with actual optimization call)
    env = NetworkEnvironment()
    env.__dict__ = data['env']
    logger = WebKPILogger()
    logger.__dict__ = data['kpi_logger']
    
    # Run one iteration of optimization
    result = run_metaheuristic(
        env=env,
        algorithm=data.get('algorithm', 'pfo'),
        epoch=data['progress']['iteration'],
        kpi_logger=logger,
        visualize_callback=None
    )
    
    # Update progress
    data['progress']['iteration'] += 1
    data['progress']['metrics'] = result['metrics']
    data['progress']['positions'] = result.get('agents', {}).get('positions', [])
    
    # Create figures
    df = pd.DataFrame(logger.history)
    fitness_fig = px.line(df, y='fitness', title="Fitness Evolution")
    sinr_fig = px.line(df, y='average_sinr', title="SINR Evolution")
    fairness_fig = px.line(df, y='fairness', title="Fairness Evolution")
    
    # Create network map
    bs_pos = env.base_stations[:, :2]
    ue_pos = env.users[:, :2]
    map_fig = px.scatter_mapbox(
        pd.DataFrame({
            'lat': np.concatenate([bs_pos[:,1], ue_pos[:,1]]),
            'lon': np.concatenate([bs_pos[:,0], ue_pos[:,0]]),
            'type': ['BS']*len(bs_pos) + ['UE']*len(ue_pos)
        }),
        lat="lat",
        lon="lon",
        color="type",
        zoom=10
    )
    map_fig.update_layout(mapbox_style="open-street-map")
    
    # Create KPI cards
    cards = [
        html.Div([
            html.H4("Fitness"),
            html.H2(f"{data['progress']['metrics'].get('fitness', 0):.2f}")
        ], style=styles['metric-card']),
        html.Div([
            html.H4("SINR (dB)"),
            html.H2(f"{data['progress']['metrics'].get('average_sinr', 0):.2f}")
        ], style=styles['metric-card']),
        html.Div([
            html.H4("Fairness"),
            html.H2(f"{data['progress']['metrics'].get('fairness', 0):.2f}")
        ], style=styles['metric-card'])
    ]
    
    return cards, fitness_fig, sinr_fig, fairness_fig, map_fig, data

@callback(
    Output('run-button', 'disabled', allow_duplicate=True),
    Input('session-data', 'data'),
    prevent_initial_call=True
)
def enable_button(data):
    return not data.get('running', False)

if __name__ == '__main__':
    app.run(debug=True)