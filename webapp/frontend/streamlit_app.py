
import streamlit as st
import sys, os, threading, time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page config ---
st.set_page_config(page_title="6G Metaheuristic Dashboard", layout="wide")

# --- Session-state initialization ---
st.session_state.setdefault("cmp", {})
for cnt in ("kpi_count","live_count","final_count","topo_single_count"):  
    st.session_state.setdefault(cnt, 0)

# Project path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

from core.envs.custom_channel_env import NetworkEnvironment
from core.hybrid_trainer.metaheuristic_opt import run_metaheuristic
from core.hybrid_trainer.kpi_logger import KPITracker

st.title("6G Metaheuristic Dashboard")

# Sidebar controls
with st.sidebar:
    mode   = st.radio("Mode", ["Single","Comparison"] )
    num_bs = st.slider("Base Stations", 5,50,10)
    num_ue = st.slider("Users", 20,200,50)
    if mode=="Single":
        algorithm = st.selectbox("Algorithm", ["pfo","pso","aco","gwo"] )
    else:
        algorithm = st.multiselect("Compare Algos", ["pfo","pso","aco","gwo"], default=["pso","aco"] )
        kpi_to_compare = st.selectbox("KPI to Compare", ["fitness","average_sinr","fairness"], index=0)
    run = st.button("Start")

# Helper to clear & plot with unique key
def clear_and_plot(ph, fig, counter_name):
    ph.empty()
    st.session_state[counter_name] += 1
    ph.plotly_chart(fig, use_container_width=True, key=f"{counter_name}_{st.session_state[counter_name]}")

# --- Always-visible Network Topology (full-width row) ---
st.session_state.topo_env = NetworkEnvironment({"num_bs":num_bs, "num_ue":num_ue}, log_kpis=False)
st.markdown("---")
ph_topo = st.expander("Network Topology", expanded=True).empty()
bs = np.array([b.position for b in st.session_state.topo_env.base_stations])
ue = np.array([u.position for u in st.session_state.topo_env.ues])
init_topo = go.Figure()
init_topo.add_trace(go.Scatter(x=bs[:,0], y=bs[:,1], mode='markers', name='BS'))
init_topo.add_trace(go.Scatter(x=ue[:,0], y=ue[:,1], mode='markers', name='UE'))
ph_topo.plotly_chart(init_topo, use_container_width=True, key="topo_init")
st.markdown("---")

# --- SINGLE MODE: use callback for real-time updates ---
if run and mode=="Single":
    ph_kpi = st.empty()
    tracker = KPITracker()
    env = st.session_state.topo_env

    def visualize(metrics, solution):
        hist = tracker.history
        if not hist.empty:
            fig_kpi = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                    subplot_titles=["Fitness","SINR","Fairness"])
            fig_kpi.add_trace(go.Scatter(x=hist.index, y=hist['fitness'], name='Fitness'), row=1, col=1)
            fig_kpi.add_trace(go.Scatter(x=hist.index, y=hist['average_sinr'], name='SINR'), row=2, col=1)
            fig_kpi.add_trace(go.Scatter(x=hist.index, y=hist['fairness'], name='Fairness'), row=3, col=1)
            clear_and_plot(ph_kpi, fig_kpi, "kpi_count")

    res = run_metaheuristic(env=env, algorithm=algorithm, epoch=0,
                             kpi_logger=tracker, visualize_callback=visualize)
    st.success("Single optimization complete!")
    st.write("Solution:", res['solution'])
    final_vals = tracker.history.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Fitness", f"{final_vals['fitness']:.3f}")
    col2.metric("Average SINR", f"{final_vals['average_sinr']:.3f}")
    col3.metric("Fairness", f"{final_vals['fairness']:.3f}")

# --- COMPARISON MODE: parallel threads & placeholder loop ---
if run and mode=="Comparison":
    ph_live_title = st.empty()
    ph_live_chart = st.empty()
    ph_live_title.subheader(f"Comparison: Live {kpi_to_compare.replace('_',' ').title()}")

    # prepare thread-safe result container
    results = {}
    st.session_state.cmp.clear()
    for alg in algorithm:
        tracker = KPITracker()
        env = NetworkEnvironment({"num_bs":num_bs,"num_ue":num_ue}, log_kpis=False)
        st.session_state.cmp[alg] = {"tracker":tracker}
        def worker(a=alg, tr=tracker, e=env):
            r = run_metaheuristic(env=e, algorithm=a, epoch=0,
                                  kpi_logger=tr, visualize_callback=None)
            results[a] = r
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        st.session_state.cmp[alg]["thread"] = t

if mode=="Comparison" and st.session_state.cmp:
    threads = [d["thread"] for d in st.session_state.cmp.values()]
    while any(t.is_alive() for t in threads):
        fig_live = make_subplots(rows=1, cols=1)
        for alg,d in st.session_state.cmp.items():
            hist = d["tracker"].history
            if not hist.empty:
                fig_live.add_trace(go.Scatter(x=hist.index, y=hist[kpi_to_compare], name=alg))
        clear_and_plot(ph_live_chart, fig_live, "live_count")
        time.sleep(1)

    # final comparison side-by-side
    # copy results into session_state
    for a, r in results.items():
        st.session_state.cmp[a]["result"] = r

    # build final bar and analysis columns
    col_bar, col_space, col_analysis = st.columns([4,1,3])
    # bar chart
    with col_bar:
        st.subheader(f"Final {kpi_to_compare.replace('_',' ').title()}")
        final_list = [{"algorithm":a, kpi_to_compare: r["metrics"][kpi_to_compare]} for a,r in results.items()]
        df = pd.DataFrame(final_list).set_index("algorithm")
        bar_fig = go.Figure(data=[go.Bar(x=df.index, y=df[kpi_to_compare])])
        st.plotly_chart(bar_fig, use_container_width=True)
    # analysis cards
    with col_analysis:
        st.subheader("Final KPI Summary")
        # for a in df.index:
        #     val = df.loc[a, kpi_to_compare]
        #     # eye-catching algorithm label
        #     st.markdown(f"**<span style='font-size:24px;'>{a.upper()}</span>** **{val:.3f}**", unsafe_allow_html=True)
        # st.markdown("**Final KPI Summary**")
        # stack each algo row: name then value
        for item in final_list:
            algo_name = item['algorithm'].upper()
            algo_val = item[kpi_to_compare]
            st.markdown(f"### {algo_name}")
            st.write(f"{algo_val:.3f}")
    st.success("Comparison complete!")







# import streamlit as st
# st.set_page_config(page_title="6G Metaheuristic Dashboard", layout="wide")

# if "single" not in st.session_state:
#     st.session_state.single = {"env":None, "tracker":None, "thread":None, "result":None}
# if "cmp" not in st.session_state:
#     st.session_state.cmp = {}
    
# import sys, os, threading, time
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# # Project path setup (optional if installed as module)
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# sys.path.insert(0, project_root) if project_root not in sys.path else None

# from core.envs.custom_channel_env import NetworkEnvironment
# from core.hybrid_trainer.metaheuristic_opt import run_metaheuristic
# from core.hybrid_trainer.kpi_logger import KPITracker


# st.title("6G Metaheuristic Dashboard")

# # ─── Initialize session_state defaults ─────────────────────────────────────

# # counters for unique keys
# for cnt in ("kpi_count","topo_count","live_count","final_count"):
#     if cnt not in st.session_state:
#         st.session_state[cnt] = 0
# # ─────────────────────────────────────────────────────────────────────────────

# # ─── Placeholders ───────────────────────────────────────────────────────────
# ph_kpi   = st.empty()
# ph_topo  = st.empty()
# ph_live  = st.empty()
# ph_final = st.empty()
# # ─────────────────────────────────────────────────────────────────────────────

# def clear_and_plot(ph, fig, counter_name):
#     """
#     Clear placeholder and plot fig with a unique key using counter_name.
#     """
#     ph.empty()
#     st.session_state[counter_name] += 1
#     ph.plotly_chart(fig, use_container_width=True, key=f"{counter_name}_{st.session_state[counter_name]}")

# # ─── Sidebar ────────────────────────────────────────────────────────────────
# with st.sidebar:
#     mode   = st.radio("Mode", ["Single","Comparison"] )
#     num_bs = st.slider("Base Stations", 5,50,10)
#     num_ue = st.slider("Users", 20,200,50)
#     if mode=="Single":
#         algorithm = st.selectbox("Algorithm", ["pfo","pso","aco","gwo"] )
#     else:
#         algorithm = st.multiselect("Compare Algos", ["pfo","pso","aco","gwo"], default=["pso","aco"] )
#     run = st.button("Start")
# # ─────────────────────────────────────────────────────────────────────────────

# # ─── Single Mode Launch ─────────────────────────────────────────────────────
# if run and mode=="Single":
#     # initialize environment, tracker, result
#     st.session_state.single["env"]     = NetworkEnvironment({"num_bs":num_bs,"num_ue":num_ue}, log_kpis=False)
#     st.session_state.single["tracker"] = KPITracker()
#     st.session_state.single["result"]  = None

#     def worker_single():
#         res = run_metaheuristic(
#             env=st.session_state.single["env"],
#             algorithm=algorithm,
#             epoch=0,
#             kpi_logger=st.session_state.single["tracker"],
#             visualize_callback=None
#         )
#         st.session_state.single["result"] = res

#     t = threading.Thread(target=worker_single, daemon=True)
#     t.start()
#     st.session_state.single["thread"] = t
# # ─────────────────────────────────────────────────────────────────────────────

# # ─── Single Mode Display ────────────────────────────────────────────────────
# # ─── Single Mode Display ────────────────────────────────────────────────────
# if mode == "Single" and st.session_state.single["tracker"]:
#     st.subheader("Single-Algorithm Live KPIs & Topology")

#     tracker = st.session_state.single["tracker"]
#     env     = st.session_state.single["env"]
#     result  = st.session_state.single["result"]
#     thread  = st.session_state.single["thread"]

#     while thread.is_alive():
#         hist = tracker.history
#         if not hist.empty:
#             fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
#                                 subplot_titles=["Fitness","SINR","Fairness"])
#             fig.add_trace(go.Scatter(x=hist.index, y=hist["fitness"], name="Fitness"), row=1, col=1)
#             fig.add_trace(go.Scatter(x=hist.index, y=hist["average_sinr"], name="SINR"), row=2, col=1)
#             fig.add_trace(go.Scatter(x=hist.index, y=hist["fairness"], name="Fairness"), row=3, col=1)
#             clear_and_plot(ph_kpi, fig, "kpi_count")

#         if env:
#             bs = np.array([b.position for b in env.base_stations])
#             ue = np.array([u.position for u in env.ues])
#             topo = go.Figure()
#             topo.add_trace(go.Scatter(x=bs[:,0], y=bs[:,1], mode="markers", name="BS"))
#             if result:
#                 sol = result["solution"]
#                 topo.add_trace(go.Scatter(
#                     x=ue[:,0], y=ue[:,1], mode="markers",
#                     marker=dict(color=sol, showscale=True), name="UE"
#                 ))
#             else:
#                 topo.add_trace(go.Scatter(x=ue[:,0], y=ue[:,1], mode="markers", name="UE"))
#             clear_and_plot(ph_topo, topo, "topo_count")

#         time.sleep(1)  # refresh every second

#     st.success("Single run complete!")
#     if result:
#         st.write("Final Solution:", result["solution"])

# # ─────────────────────────────────────────────────────────────────────────────

# # ─── Comparison Mode Launch ─────────────────────────────────────────────────
# if run and mode=="Comparison":
#     st.session_state.cmp.clear()
#     for alg in algorithm:
#         tracker = KPITracker()
#         env     = NetworkEnvironment({"num_bs":num_bs,"num_ue":num_ue}, log_kpis=False)
#         st.session_state.cmp[alg] = {"tracker":tracker, "thread":None, "result":None}
#         def make_worker(a=alg,tr=tracker,en=env):
#             res = run_metaheuristic(env=en, algorithm=a, epoch=0,
#                                     kpi_logger=tr, visualize_callback=None)
#             st.session_state.cmp[a]["result"] = res
#         t = threading.Thread(target=make_worker, daemon=True)
#         t.start()
#         st.session_state.cmp[alg]["thread"] = t
# # ─────────────────────────────────────────────────────────────────────────────

# # ─── Comparison Mode Display ────────────────────────────────────────────────
# if mode=="Comparison" and st.session_state.cmp:
#     st.subheader("Comparison: Live Fitness Curves")
#     threads = [d["thread"] for d in st.session_state.cmp.values()]
#     # update while any thread alive
#     while any(t.is_alive() for t in threads):
#         fig = make_subplots(rows=1, cols=1)
#         for alg,d in st.session_state.cmp.items():
#             hist = d["tracker"].history
#             if not hist.empty:
#                 fig.add_trace(go.Scatter(x=hist.index, y=hist["fitness"], name=alg))
#         clear_and_plot(ph_live, fig, "live_count")
#         time.sleep(1)

#     # final bar chart
#     st.subheader("Comparison: Final Fitness")
#     res = []
#     for alg,d in st.session_state.cmp.items():
#         if d["result"]:
#             res.append({"algorithm":alg,"fitness":d["result"]["metrics"]["fitness"]})
#     if res:
#         df = pd.DataFrame(res).set_index("algorithm")
#         bar = go.Figure(data=[go.Bar(x=df.index, y=df["fitness"])])
#         clear_and_plot(ph_final, bar, "final_count")
#         st.success("Comparison complete!")






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