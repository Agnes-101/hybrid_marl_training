# streamlit_app.py (modified)
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np

# Path configuration
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root) if project_root not in sys.path else None

from core.envs.custom_channel_env import NetworkEnvironment
from core.hybrid_trainer.metaheuristic_opt import run_metaheuristic
from core.hybrid_trainer.kpi_logger import KPITracker

def display_kpis(metrics):
    """Render KPIs in separate columns with individual line charts"""
    if not metrics or 'history' not in metrics:
        st.warning("No metrics data available yet")
        return

    history_df = pd.DataFrame(metrics['history'])
    
    st.subheader("📊 KPI Evolution")

    st.markdown("### Fitness Evolution")
    st.metric("Current Fitness", f"{metrics.get('fitness', 0):.2f}")
    st.line_chart(history_df[['fitness']])

    st.markdown("### SINR Evolution")
    st.metric("Average SINR", f"{metrics.get('average_sinr', 0):.2f} dB")
    st.line_chart(history_df[['average_sinr']])

    st.markdown("### Fairness Evolution")
    st.metric("Fairness Index", f"{metrics.get('fairness', 0):.2f}")
    st.line_chart(history_df[['fairness']])

    with st.expander("📈 Combined Line Chart", expanded=False):
        if not history_df.empty:
            st.line_chart(history_df[['fitness', 'average_sinr', 'fairness']])
    # # Create 3 columns for metrics
    # col1, col2, col3 = st.rows(3)
    
    # with col1:
    #     st.markdown("**Fitness Evolution**")
    #     st.metric("Current Fitness", f"{metrics.get('fitness', 0):.2f}")
    #     st.line_chart(
    #         history_df[['fitness']], 
    #         use_container_width=True,
    #         color="#FF4B4B"  # Streamlit's red color
    #     )
    
    # with col2:
    #     st.markdown("**SINR Evolution**")
    #     st.metric("Average SINR", f"{metrics.get('average_sinr', 0):.2f} dB")
    #     st.line_chart(
    #         history_df[['average_sinr']], 
    #         use_container_width=True,
    #         color="#00CCCC"  # Streamlit's teal color
    #     )
    
    # with col3:
    #     st.markdown("**Fairness Evolution**")
    #     st.metric("Fairness Index", f"{metrics.get('fairness', 0):.2f}")
    #     st.line_chart(
    #         history_df[['fairness']], 
    #         use_container_width=True,
    #         color="#00AA00"  # Green color
    #     )
    
    # with st.expander("Performance Metrics", expanded=True):
    #     cols = st.columns(3)
    #     cols[0].metric("Current Fitness", f"{metrics.get('fitness', 0):.2f}")
    #     cols[1].metric("Avg SINR", f"{metrics.get('average_sinr', 0):.2f} dB")
    #     cols[2].metric("Fairness", f"{metrics.get('fairness', 0):.2f}")
        
    #     if not df.empty:
    #         st.line_chart(df[['fitness', 'average_sinr', 'fairness']])

def display_positions(solution, env):
    """Show BS/UE positions with solution mapping"""
    try:
        # Get positions from environment
        bs_pos = env.base_stations[:, :2]  # First 2 coordinates
        ue_pos = env.users[:, :2]
        
        # Create DataFrame
        df = pd.DataFrame(
            np.vstack([bs_pos, ue_pos]),
            columns=['x', 'y']
        )
        df['type'] = ['BS']*len(bs_pos) + ['UE']*len(ue_pos)
        
        # Add solution coloring
        df['cluster'] = [None]*len(bs_pos) + list(solution)
        
        st.map(df, latitude='y', longitude='x', color='cluster', size=15)
    except Exception as e:
        st.error(f"Position error: {str(e)}")

def main():
    st.title("6G MARL Optimization (Direct Mode)")
    
    # Control Panel
    with st.sidebar:
        st.header("Configuration")
        algorithm = st.selectbox("Algorithm", ["pfo", "pso", "aco"], index=0)
        num_bs = st.slider("Base Stations", 5, 50, 10)
        num_ue = st.slider("Users", 20, 200, 50)
        
        if st.button("Run Optimization"):
            st.session_state.result = None
            st.session_state.env = None
            with st.spinner("Optimizing..."):
                try:
                    # Initialize environment and logger
                    # Set up environment
                    env_config = {
                        "num_ue": num_ue,
                        "num_bs": num_bs
                    }
                    env = NetworkEnvironment(config=env_config, log_kpis=False)

                    kpi_logger = KPITracker()
                    
                    # Run optimization
                    result = run_metaheuristic(
                        env=env,
                        algorithm=algorithm,
                        epoch=0,
                        kpi_logger=kpi_logger,
                        visualize_callback=None
                    )
                    
                    # Store results with full metrics
                    st.session_state.result = {
                        "solution": result["solution"],
                        "metrics": {
                            **result["metrics"],
                            "history": kpi_logger.history.to_dict()
                        },
                        "env": env
                    }
                except Exception as e:
                    st.error(f"Optimization failed: {str(e)}")

    # Display results
    if st.session_state.get("result"):
        result = st.session_state.result
        display_kpis(result["metrics"])
        display_positions(result["solution"], result["env"])

if __name__ == "__main__":
    main()


# # streamlit_app.py
# import streamlit as st
# import requests
# import pandas as pd
# import time
# from typing import Optional

# # Configuration
# BACKEND_URL = "http://localhost:8000"  # FastAPI endpoint
# POLL_INTERVAL = 1.5  # Seconds between updates

# def get_task_status(task_id: str) -> Optional[dict]:
#     """Fetch current task state from backend"""
#     try:
#         response = requests.get(f"{BACKEND_URL}/status/{task_id}")
        

#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error fetching task status: {str(e)}")
#         return None

# # def display_kpis(metrics: dict):
# #     """Render KPI metrics using Streamlit components"""
# #     with st.expander("Live Metrics", expanded=True):
# #         col1, col2, col3 = st.columns(3)
# #         with col1:
# #             st.metric("Current Fitness", f"{metrics.get('fitness', 0):.2f}")
# #             st.line_chart(pd.DataFrame(metrics['history'])[['fitness']])
# #         with col2:
# #             st.metric("Average SINR", f"{metrics.get('average_sinr', 0):.2f} dB")
# #             st.line_chart(pd.DataFrame(metrics['history'])[['average_sinr']])
# #         with col3:
# #             st.metric("Fairness", f"{metrics.get('fairness', 0):.2f}")
# #             st.line_chart(pd.DataFrame(metrics['history'])[['fairness']])
# def display_kpis(metrics):
#     """Handle missing data gracefully"""
#     if not metrics or 'history' not in metrics:
#         st.warning("No metrics data available yet")
#         return

#     # Safely get values with defaults
#     current_fitness = metrics.get('fitness', 0)
#     history_df = pd.DataFrame(metrics['history'])
    
#     with st.expander("Live Metrics", expanded=True):
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("Current Fitness", f"{current_fitness:.2f}")
#             st.line_chart(history_df[['fitness']])
#         with col2:
#             st.metric("Average SINR", f"{metrics.get('average_sinr', 0):.2f} dB")
#             st.line_chart(history_df[['average_sinr']])
#         with col3:
#             st.metric("Fairness", f"{metrics.get('fairness', 0):.2f}")
#             st.line_chart(history_df[['fairness']])

# def display_positions(positions: list):
#     """Show agent positions on 2D plane"""
#     if positions:
#         try:
#             df = pd.DataFrame(positions, columns=['x', 'y'])
#             st.map(df.assign(lat=df['y'], lon=df['x']))  # Streamlit's map expects lat/lon
#         except Exception as e:
#             st.error(f"Position visualization error: {str(e)}")

# def main():
#     st.title("6G MARL Optimization Monitor")
    
#     # Initialize session state
#     if 'task_id' not in st.session_state:
#         st.session_state.task_id = None
#     if 'running' not in st.session_state:
#         st.session_state.running = False

#     # Control Panel (Sidebar)
#     with st.sidebar:
#         st.header("Optimization Controls")
        
#         # Algorithm selection
#         algorithm = st.selectbox(
#             "Optimization Algorithm",
#             options=["pfo", "pso", "aco", "gwo"],
#             index=0
#         )
        
#         # Environment parameters
#         num_bs = st.slider("Base Stations", 1, 100, 10)
#         num_ue = st.slider("User Equipment", 10, 300, 50)
        
#         # Start/Stop controls
#         if st.button("Start Optimization") and not st.session_state.running:
#             try:
#                 response = requests.post(
#                     f"{BACKEND_URL}/start",
#                     json={
#                         "algorithm": algorithm,
#                         "num_bs": num_bs,
#                         "num_ue": num_ue
#                     }
#                 )
#                 response.raise_for_status()
#                 st.session_state.task_id = response.json()["task_id"]
#                 st.session_state.running = True
#                 st.success("Optimization started successfully!")
#             except Exception as e:
#                 st.error(f"Failed to start task: {str(e)}")

#     # Main Visualization Area
#     if st.session_state.running and st.session_state.task_id:
#         status = get_task_status(st.session_state.task_id)
        
#         if not status:
#             st.session_state.running = False
#             return
        
#         # status_col, = st.columns(1)
#         # with status_col:
#         #     st.subheader("Task Status")
#         #     st.write(f"**State**: {status.get('status', 'unknown')}")
            
#         #     if status['status'] == 'PROGRESS':
#         #         st.write(f"**Iteration**: {status.get('progress', {}).get('metrics', {}).get('iteration', 0)}")
#         #         display_kpis(status.get('progress', {}).get('metrics', {}))
#         #         display_positions(status.get('positions', []))
#         #         # st.experimental_rerun()  # Auto-refresh
#         #         time.sleep(POLL_INTERVAL)
#         #         st.rerun()
#         #         # meta = status.get('meta', {})
                
#         #     elif status['status'] == 'SUCCESS':
#         #         st.balloons()
#         #         display_kpis(status.get('result', {}).get('metrics', {}))
#         #         display_positions(status.get('result', {}).get('positions', []))
                
#         #     elif status['status'] == 'FAILURE':
#         #         st.error(f"Task failed: {status.get('message', 'Unknown error')}")
#         # # Handle different task states
#         # if status['status'] == 'SUCCESS':
#         #     st.balloons()
#         #     st.success("Optimization completed!")
#         #     st.session_state.running = False
#         #     display_kpis(status['result'])
#         #     return
            
#         # elif status['status'] == 'FAILURE':
#         #     st.error(f"Task failed: {status.get('error', 'Unknown error')}")
#         #     st.session_state.running = False
#         #     return
            
#         # elif status['status'] == 'PROGRESS':
#         #     # Extract metrics from task metadata
#         #     meta = status.get('meta', {})
            
#             # if meta.get('type') == 'KPI_BATCH':
#             #     # Display latest metrics
#             #     display_kpis({
#             #         'fitness': meta['data']['summary']['avg_fitness'],
#             #         'average_sinr': meta['data']['summary']['max_sinr'],
#             #         'fairness': meta['data']['batch'][-1]['fairness'],
#             #         'history': meta['data']['batch']
#             #     })
                
#             #     # Display agent positions if available
#             #     if 'agents' in meta:
#             #         display_positions(meta['agents']['positions'])
            
#             # Create manual refresh button
#             if st.button("Force Refresh"):
#                 st.rerun()
            
#             # # Auto-refresh every POLL_INTERVAL seconds
#             # time.sleep(POLL_INTERVAL)
#             # st.rerun()

# if __name__ == "__main__":
#     main()