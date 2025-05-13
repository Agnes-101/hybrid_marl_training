import asyncio
# Ensure there's an active event loop for Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import sys
import os
import threading
import time
import base64
import copy
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Load BS icon
icon_path = os.path.join(os.path.dirname(__file__), "assets", "bs_icon.png")
with open(icon_path, "rb") as f:
    bs_b64 = base64.b64encode(f.read()).decode()
# # # --- Algorithm info folder setup ---
# # algo_info_dir = os.path.join(os.path.dirname(__file__), "algo_info")  # markdown files here
# # algo_image_dir = os.path.join(os.path.dirname(__file__), "assets", "algo_images")  # images per algorithm

# Algorithm info
import json
info_path = os.path.join(os.path.dirname(__file__), "assets", "algo_info.json")
with open(info_path) as f:
    algo_info = json.load(f)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

from core.envs.custom_channel_env import NetworkEnvironment
from core.hybrid_trainer.metaheuristic_opt import run_metaheuristic
from core.hybrid_trainer.kpi_logger import KPITracker
from core.hybrid_trainer.marl_runner import run_marl

st.set_page_config(page_title="6G Metaheuristic & MARL Dashboard", layout="wide")
st.title("6G Metaheuristic & MARL Dashboard")

# Session-state for flicker control
if "figures" not in st.session_state:
    st.session_state.figures = {}
for cnt in ("kpi_count", "live_count", "final_count", "topo_count", "progress_count","viz_counter"):
    st.session_state.setdefault(cnt, 0)
st.session_state.setdefault("hybrid_phase", "meta")
st.session_state.setdefault("sol0", None)
st.session_state.setdefault("hybrid_thread_started", False)
st.session_state.setdefault("hybrid_done", False)
# Add these to your session state initialization section
st.session_state.setdefault("current_step", 0)
st.session_state.setdefault("latest_solution", None)
st.session_state.setdefault("meta_result", None)
st.session_state.setdefault("latest_metrics", {})
st.session_state.setdefault("progress_value", 0)
st.session_state.setdefault("last_kpi_step", -1)
st.session_state.setdefault("last_viz_trigger", -1)

# --- Sidebar ---
with st.sidebar:
    mode = st.radio("Mode", ["Single Metaheuristic", "Comparison", "MARL", "Hybrid"])
    num_bs = st.slider("Base Stations", 5, 50, 10)
    num_ue = st.slider("Users", 20, 500, 50)
    if mode in ["Single Metaheuristic", "Hybrid"]:
        meta_algo = st.selectbox("Metaheuristic Algorithm", ["pfo", "co", "coa", "do", "fla", "gto", "hba", "hoa", "avoa","aqua", "poa", "rime", "roa", "rsa", "sto"])
    if mode == "Comparison":
        algos = st.multiselect("Compare Algos", ["avoa", "aqua","co", "coa", "do", "fla", "gto", "hba", "hoa", "pfo", "poa", "rime", "roa", "rsa", "sto"], default=["pfo", "co"])
        kpi_cmp = st.selectbox("KPI to Compare", ["fitness", "average_sinr", "fairness"])
    if mode in ["MARL", "Hybrid"]:
        marl_steps = st.slider("MARL Steps/Epoch", 1, 50, 10)
        # Add visualization frequency control
        viz_freq = st.slider("Visualization Frequency", 1, 20, 5, 
                             help="Update topology visualization every N steps (higher = faster but less visual feedback)")
    # run = st.button("Start")
    if st.button("Start"):
        st.session_state.started = True

    run = st.session_state.get("started", False)


# flicker-free plot helper
def clear_and_plot(ph, fig, key):
    figs = st.session_state["figures"] 
    if key in figs and figs[key] is not None:
        st.session_state[key] += 1
        figs[key] = copy.deepcopy(fig)
        ph.plotly_chart(fig, use_container_width=True, key=f"{key}_{st.session_state[key]}")
    else:
        ph.empty()
        st.session_state[key] += 1
        ph.plotly_chart(fig, use_container_width=True, key=f"{key}_{st.session_state[key]}")
        figs[key] = fig

# topology drawer
def render_topology(env, solution=None):
    bs_coords = np.array([b.position for b in env.base_stations])
    ue_coords = np.array([u.position for u in env.ues])
    
    fig = go.Figure()
    
    # BS markers + icons
    fig.add_trace(go.Scatter(
        x=bs_coords[:, 0], 
        y=bs_coords[:, 1], 
        mode='markers', 
        name='BS', 
        marker=dict(size=0, color='rgba(0,0,0,0)'), 
        showlegend=True
    ))
    
    for x, y in bs_coords:
        fig.add_layout_image(
            source=f"data:image/png;base64,{bs_b64}", 
            xref="x", 
            yref="y", 
            x=x, 
            y=y,
            sizex=7, 
            sizey=7, 
            xanchor="center", 
            yanchor="middle", 
            layer="above"
        )
    
    # If solution is provided, draw connection lines
    if solution is not None and len(solution) > 0:
        for i, (ux, uy) in enumerate(ue_coords):
            bs_id = solution[i]
            bx, by = bs_coords[bs_id]
            fig.add_trace(go.Scatter(
                x=[bx, ux], 
                y=[by, uy], 
                mode='lines', 
                line=dict(color='lightgray', width=1), 
                showlegend=False
            ))
        # Add UEs with connection info
        custom = np.stack([np.arange(len(ue_coords)), solution], axis=1)
    else:
        # Add UEs without connection info
        custom = np.stack([np.arange(len(ue_coords)), [-1]*len(ue_coords)], axis=1)
    
    # Add UE markers with customdata
    fig.add_trace(go.Scatter(
        x=ue_coords[:, 0], 
        y=ue_coords[:, 1], 
        mode='markers', 
        name='UE', 
        marker=dict(size=10),
        hovertemplate="UE %{customdata[0]}<br>Assigned BS %{customdata[1]}<extra></extra>",
        customdata=custom
    ))
    
    fig.update_layout(
        title="Network Topology", 
        xaxis=dict(scaleanchor="y"), 
        yaxis=dict(constrain="domain")
    )
    
    return fig

# layout containers
col1, col2 = st.columns([3, 1])
with col1:
    ph_topo = st.expander("Network Topology", expanded=True).empty()
with col2:
    # st.markdown("### Algorithm Info")
    # if mode in ["Single Metaheuristic", "Hybrid"]:
    #     info = algo_info.get(meta_algo, {})
    #     st.write(f"**{info.get('name', meta_algo)}**")
    #     st.write(info.get("short", ""))
    # elif mode == "MARL":
    #     info = algo_info.get("marl", {})
    #     st.write("**MARL (PPO)**")
    #     st.write(info.get("short", ""))
    # else:
    #     for a in algos:
    #         info = algo_info.get(a, {})
    #         st.write(f"**{info.get('name', a)}**: {info.get('short', '')}")
    # In your layout containers section, replace the expander block with:

    with st.expander("Algorithm Info", expanded=True):
        if mode in ["Single Metaheuristic", "Hybrid"]:
            info = algo_info.get(meta_algo.lower(), {})
            if info:
            
                st.markdown(f"### {info.get('name', meta_algo)}")
                try:
                    image_path = os.path.join(os.path.dirname(__file__), info["image"])
                    st.image(image_path, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
                st.markdown(f"**{info.get('short', '')}**")
                st.markdown("#### Key Operations:")
                st.markdown(info.get("long", "No detailed description available."))
            else:
                st.warning(f"No information found for {meta_algo}")
        elif mode == "Comparison":
            for algo in algos:
                info = algo_info.get(algo.lower(), {})
                
                
                if info:
                    st.markdown(f"**{info.get('name', algo)}**: {info.get('short','')}")
                    # with st.markdown(f"{info.get('name', algo)}"):
                    #     cols = st.columns([3, 1])
                    #     cols[0].markdown(f"**{info.get('short', '')}**")
                        #cols[0].markdown("**Key Operations:**")
                        # cols[0].markdown(info.get("long", ""))
                        # try:
                        #     image_path = os.path.join(os.path.dirname(__file__), info["image"])
                        #     cols[1].image(image_path, use_container_width=True)
                        # except Exception as e:
                        #     cols[1].error(f"Image error: {str(e)}")
                else:
                    st.warning(f"No information found for {algo}")    
                    
# Initialize environment and display initial network topology
env = NetworkEnvironment({"num_bs": num_bs, "num_ue": num_ue, "episode_length": 100}, log_kpis=False)
initial_topo = render_topology(env)
ph_topo.plotly_chart(initial_topo, use_container_width=True)

# --- Main logic ---
if run:
    # Single Metaheuristic
    if mode == "Single Metaheuristic":
        tracker = KPITracker()
        ph_kpi = st.empty()
        
        def viz(metrics, solution):
            h = tracker.history
            if not h.empty:
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["Fitness", "SINR", "Fairness"])
                fig.update_layout(height=600, margin=dict(t=50, b=40, l=40, r=40))
                for i in (1, 2, 3): 
                    fig.update_yaxes(showgrid=True, gridwidth=1, row=i, col=1)
                    fig.update_xaxes(showgrid=True, gridwidth=1, row=i, col=1)
                fig.add_trace(go.Scatter(x=h.index, y=h.fitness, name="Fitness", line=dict(width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=h.index, y=h.average_sinr, name="SINR", line=dict(width=2)), row=2, col=1)
                fig.add_trace(go.Scatter(x=h.index, y=h.fairness, name="Fairness", line=dict(width=2)), row=3, col=1)
                clear_and_plot(ph_kpi, fig, "kpi_count")
            
            topo_fig = render_topology(env, solution)
            clear_and_plot(ph_topo, topo_fig, "topo_count")
            
        result = run_metaheuristic(env, meta_algo, epoch=0, kpi_logger=tracker, visualize_callback=viz)
        
        # Final render to ensure visibility
        if result and 'solution' in result:
            final_topo = render_topology(env, result['solution'])
            clear_and_plot(ph_topo, final_topo, "topo_count")
            
            # Display final KPIs
            metrics = result["metrics"]
            st.markdown("### Final KPIs Summary")
            kpi_labels = {
                "fitness": ("Fitness Score", ""),
                "average_sinr": ("Avg SINR", "dB"),
                "fairness": ("Fairness Index", "")
            }
            cols = st.columns(3)
            for i, (kpi, (label, unit)) in enumerate(kpi_labels.items()):
                val = metrics.get(kpi, 0)
                cols[i].metric(label, f"{val:.3f} {unit}")
                
        st.success("Metaheuristic done")

    # Comparison
    elif mode == "Comparison":
        ph_live_title = st.empty()
        ph_live_chart = st.empty()
        ph_live_title.subheader(f"Comparison: Live {kpi_cmp.replace('_', ' ').title()}")
        
        trackers = {}
        threads = {}
        results = {}
        env_dict = {}
        
        for a in algos:
            tr = KPITracker()
            trackers[a] = tr
            e = NetworkEnvironment({"num_bs": num_bs, "num_ue": num_ue}, log_kpis=False)
            env_dict[a] = e
            
            def w(a=a, e=e, tr=tr): 
                results[a] = run_metaheuristic(e, a, 0, tr, None)
                
            t = threading.Thread(target=w, daemon=True)
            threads[a] = t
            t.start()
            
        while any(t.is_alive() for t in threads.values()):
            fig_live = make_subplots(rows=1, cols=1)
            for a, tr in trackers.items():
                h = tr.history
                if not h.empty and kpi_cmp in h:
                    fig_live.add_trace(go.Scatter(x=h.index, y=h[kpi_cmp], name=a))
            clear_and_plot(ph_live_chart, fig_live, "live_count")
            time.sleep(1)
            
        # After comparison is done, find the best algorithm
        best_algo = None
        best_value = float('-inf') if kpi_cmp != 'fairness' else 0
        
        for algo, result in results.items():
            if algo in trackers and not trackers[algo].history.empty:
                final_value = trackers[algo].history[kpi_cmp].iloc[-1]
                if final_value > best_value:
                    best_value = final_value
                    best_algo = algo
        
        # Display final results
        c1, _, c3 = st.columns([4, 1, 3])
        with c1:
            st.subheader(f"Final {kpi_cmp.replace('_', ' ').title()}")
            df = pd.DataFrame([{"alg": a, kpi_cmp: r["metrics"][kpi_cmp]} for a, r in results.items()]).set_index("alg")
            st.plotly_chart(go.Figure(data=[go.Bar(x=df.index, y=df[kpi_cmp])]), use_container_width=True)
        
        with c3:
            st.subheader("Final KPI Summary")
            for a, r in results.items():
                st.markdown(f"### {a.upper()}")
                st.write(f"{r['metrics'][kpi_cmp]:.3f}")
        
        # Show the best algorithm's topology
        if best_algo and best_algo in results and 'solution' in results[best_algo]:
            st.write(f"Showing topology for best algorithm: **{best_algo}** (best {kpi_cmp}: {best_value:.4f})")
            best_topo = render_topology(env_dict[best_algo], results[best_algo]['solution'])
            clear_and_plot(ph_topo, best_topo, "topo_count")
            
        st.success("Comparison done")

    # MARL
    elif mode == "MARL":
        ph_kpi = st.empty()
        ph_progress = st.empty()
        tracker = KPITracker()
        
        env_cfg = {"num_bs": num_bs, "num_ue": num_ue, "episode_length": 100}
        ray_res = {"num_cpus": 4, "num_gpus": 0}
        
        final_sol = None
        step_counter = 0
        total_epochs = 50
        
        # Create a progress bar
        progress_bar = ph_progress.progress(0)
        
        for metrics, sol in run_marl(env_cfg, ray_res, None, marl_steps, total_epochs=total_epochs):
            step_counter += 1
            
            # Update KPI chart on every step
            h = tracker.history
            if not h.empty:
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["Mean Reward", "Min Reward", "Max Reward"])
                fig.update_layout(height=600, margin=dict(t=50, b=40, l=40, r=40))
                for i in (1, 2, 3): 
                    fig.update_yaxes(showgrid=True, gridwidth=1, row=i, col=1)
                    fig.update_xaxes(showgrid=True, gridwidth=1, row=i, col=1)
                fig.add_trace(go.Scatter(x=h.index, y=h.reward_mean, name="Mean", line=dict(width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=h.index, y=h.reward_min, name="Min", line=dict(width=2)), row=2, col=1)
                fig.add_trace(go.Scatter(x=h.index, y=h.reward_max, name="Max", line=dict(width=2)), row=3, col=1)
                clear_and_plot(ph_kpi, fig, "kpi_count")
            
            # Update topology visualization only at specified frequency
            if step_counter % viz_freq == 0 or step_counter == 1:
                topo_fig = render_topology(env, sol)
                clear_and_plot(ph_topo, topo_fig, "topo_count")
            
            # Update progress bar
            progress_bar.progress(min(step_counter / total_epochs, 1.0))
            
            final_sol = sol
            
        # Final render to ensure topology is visible
        if final_sol is not None:
            final_topo = render_topology(env, final_sol)
            clear_and_plot(ph_topo, final_topo, "topo_count")
            
        st.success("MARL done")

    # --- Hybrid Mode ---
    elif mode == "Hybrid":
        ph_kpi       = st.empty()
        ph_progress  = st.empty()
        ph_status    = st.empty()
        ph_topo_meta = st.empty()
        # ph_topo already defined above
        tracker      = st.session_state.setdefault("hybrid_tracker", KPITracker())
        # define exactly the same viz function you use in Single mode:
        def viz_meta(metrics, solution):
            # 1) draw KPI history
            h = tracker.history
            if not h.empty:
                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=["Fitness", "SINR", "Fairness"]
                )
                for i, col in enumerate(["fitness","average_sinr","fairness"], start=1):
                    fig.add_trace(
                        go.Scatter(x=h.index, y=h[col], name=col.capitalize()),
                        row=i, col=1
                    )
                    fig.update_yaxes(showgrid=True, row=i, col=1)
                clear_and_plot(ph_kpi, fig, "kpi_count")

            # 2) draw topology with the intermediate solution
            topo_fig = render_topology(env, solution)
            clear_and_plot(ph_topo_meta, topo_fig, "topo_count")
            
        # 1) Metaheuristic warm-start
        if st.session_state.hybrid_phase == "meta":
            st.info(f"Running metaheuristic ({meta_algo}) for warm-start…")
            meta_result = run_metaheuristic(env, meta_algo, epoch=0, kpi_logger=tracker, visualize_callback=viz_meta)
            # meta_result = run_metaheuristic(env, meta_algo, 0, tracker, None)
            sol0 = meta_result["solution"]
            st.write(f"Initial solution from {meta_algo}")
            clear_and_plot(ph_topo, render_topology(env, sol0), "topo_count")
            time.sleep(1)
            # At the end of the metaheuristic phase:
            st.session_state.meta_result = meta_result
            # flip phase and restart script just once
            st.session_state.hybrid_phase = "marl"
            st.session_state.sol0 = sol0
            st.rerun()

        # Replace the MARL refinement phase section in the Hybrid mode with this optimized version:

        # 2) MARL refinement phase
        if st.session_state.hybrid_phase == "marl":
            st.info("Running MARL refinement…")
            env_cfg = {"num_bs": num_bs, "num_ue": num_ue, "episode_length": 100}
            ray_res = {"num_cpus": 4, "num_gpus": 0}
            total_epochs = 50

            # placeholders
            progress_bar = ph_progress.progress(0)
            initial_sol = st.session_state.get("sol0", None)
            
            # Use session state variables to control visualization frequency
            if "viz_counter" not in st.session_state:
                st.session_state.viz_counter = 0
            if "last_updated_step" not in st.session_state:
                st.session_state.last_updated_step = -1
            
            # Start background thread exactly once
            if not st.session_state.get("hybrid_thread_started", False):
                st.session_state.hybrid_thread_started = True

                def marl_bg():
                    # use the very same env object you rendered initially
                    for metrics, sol in run_marl(env_cfg, ray_res, initial_sol, marl_steps, total_epochs=total_epochs):
                        st.session_state.latest_solution = sol
                        st.session_state.progress_value = min(
                            st.session_state.get("progress_value",0) + 1/total_epochs, 1.0
                        )
                        st.session_state.viz_counter += 1
                        # feed the solution back into the env so that env.ues.associated_bs gets set
                        for ue, bs_id in zip(env.ues, sol):
                            ue.associated_bs = bs_id
                        time.sleep(0.05)
                    st.session_state.hybrid_done = True


                threading.Thread(target=marl_bg, daemon=True).start()
            
            # Update progress bar based on session state
            progress_value = st.session_state.get("progress_value", 0)
            progress_bar.progress(progress_value)
            
            # Update KPI chart based on session state
            h = tracker.history
            if not h.empty:
                # Only generate the figure if we have new data to show (avoids flickering)
                current_step = st.session_state.get("current_step", 0)
                
                # Only update the KPI chart when we have new data
                if current_step > st.session_state.get("last_kpi_step", -1):
                    st.session_state.last_kpi_step = current_step
                    
                    fig_kpi = make_subplots(
                        rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["Mean Reward", "Min Reward", "Max Reward"]
                    )
                    if "reward_mean" in h.columns:
                        fig_kpi.add_trace(go.Scatter(x=h.index, y=h.reward_mean, name="Mean"), row=1, col=1)
                        fig_kpi.add_trace(go.Scatter(x=h.index, y=h.reward_min, name="Min"), row=2, col=1)
                        fig_kpi.add_trace(go.Scatter(x=h.index, y=h.reward_max, name="Max"), row=3, col=1)
                        
                    clear_and_plot(ph_kpi, fig_kpi, "kpi_count")
            
            # Update topology visualization based on latest solution (but controlled by viz_counter)
            viz_trigger = st.session_state.get("viz_counter", 0)
            if viz_trigger > st.session_state.get("last_viz_trigger", -1) and viz_trigger % viz_freq == 0:
                st.session_state.last_viz_trigger = viz_trigger
                
                latest_sol = st.session_state.get("latest_solution")
                if latest_sol is not None:
                    # Ensure solution is valid
                    safe_sol = [bs if 0 <= bs < num_bs else 0 for bs in latest_sol]
                    
                    # Update topology visualization with the latest solution
                    topo_fig = render_topology(env, safe_sol)
                    clear_and_plot(ph_topo, topo_fig, "topo_count")
            
            # Show final results once hybrid mode is done
            if st.session_state.get("hybrid_done", False):
                latest_sol = st.session_state.get("latest_solution")
                if latest_sol is not None:
                    safe_final = [bs if 0 <= bs < num_bs else 0 for bs in latest_sol]
                    clear_and_plot(ph_topo, render_topology(env, safe_final), "topo_count")
                
                st.success("Hybrid optimization completed!")
                
                # Display meta vs. post-MARL KPIs
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**Metaheuristic KPIs**")
                    meta_result = st.session_state.get("meta_result", {})
                    metrics = meta_result.get("metrics", {})
                    for k, v in metrics.items():
                        st.write(f"{k}: {v:.3f}")
                
                with cols[1]:
                    st.markdown("**Post-MARL KPIs**")
                    latest_metrics = st.session_state.get("latest_metrics", {})
                    for k in ("reward_mean", "reward_min", "reward_max"):
                        if k in latest_metrics:
                            st.write(f"{k}: {latest_metrics[k]:.3f}")
            
            # Use a placeholder for auto-refresh without calling rerun
            # This creates a subtle auto-refresh effect while preventing memory issues
            if not st.session_state.get("hybrid_done", False):
                # Show a refreshing timestamp that changes every second
                # This tricks Streamlit into periodically updating the UI without rerun
                refresh_placeholder = st.empty()
                refresh_placeholder.markdown(f"Last update: {time.time():.0f}")
                time.sleep(0.5)  # Give UI time to update without consuming too much CPU
        

        
# import os
# # Disable Streamlit file watcher to avoid torch warnings
# os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

# import asyncio
# # Ensure there's an active event loop for Streamlit
# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())

# import streamlit as st
# import sys
# import os
# import threading
# import time
# import base64
# import copy
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# # Load BS icon
# icon_path = os.path.join(os.path.dirname(__file__), "assets", "bs_icon.png")
# with open(icon_path, "rb") as f:
#     bs_b64 = base64.b64encode(f.read()).decode()

# # Algorithm info
# import json
# info_path = os.path.join(os.path.dirname(__file__), "assets", "algo_info.json")
# with open(info_path) as f:
#     algo_info = json.load(f)

# # Add project root to path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# sys.path.insert(0, project_root)

# from core.envs.custom_channel_env import NetworkEnvironment
# from core.hybrid_trainer.metaheuristic_opt import run_metaheuristic
# from core.hybrid_trainer.kpi_logger import KPITracker
# from core.hybrid_trainer.marl_runner import run_marl

# # --- Initialize session state defaults to avoid missing-key errors ---
# def _init_session_state():
#     defaults = {
#         "run_active":        False,
#         "mode":              None,
#         "num_bs":            10,
#         "num_ue":            50,
#         "meta_algo":         "pfo",
#         "algos":             [],
#         "kpi_cmp":           "fitness",
#         "marl_steps":        10,
#         "viz_freq":          5,
#         "thread_active":     False,
#         "thread_start":      False,
#         "current_solution":  None,
#         "metrics":           {},
#         "meta_result":       None,
#         "comparison_results": None,
#         "comparison_trackers": None,
#         "comparison_envs":     None,
#         "best_algorithm":      None,
#         "step_counter":      0,
#         "total_epochs":      50,
#         "tracker":           KPITracker(),
#         "figures":           {},
#         "env":               None,
#     }
#     for k, v in defaults.items():
#         st.session_state.setdefault(k, v)

# _init_session_state()

# st.set_page_config(page_title="6G Metaheuristic & MARL Dashboard", layout="wide")
# st.title("6G Metaheuristic & MARL Dashboard")

# # For flicker control of plot updates
# for cnt in ("kpi_count", "live_count", "final_count", "topo_count", "progress_count"):
#     st.session_state.setdefault(cnt, 0)

# # --- Sidebar ---
# with st.sidebar:
#     if not st.session_state.run_active:
#         mode = st.radio("Mode", ["Single Metaheuristic", "Comparison", "MARL", "Hybrid"] , index=0)
#         num_bs = st.slider("Base Stations", 5, 50, st.session_state.num_bs)
#         num_ue = st.slider("Users", 20, 200, st.session_state.num_ue)
        
#         if mode in ["Single Metaheuristic", "Hybrid"]:
#             meta_algo = st.selectbox("Metaheuristic Algorithm", ["pfo", "co", "coa", "do", "fla", "gto", "hba", "hoa", "avoa", "poa", "rime", "roa", "rsa", "sto"], index=["pfo","co","coa","do","fla","gto","hba","hoa","avoa","poa","rime","roa","rsa","sto"].index(st.session_state.meta_algo))
#         if mode == "Comparison":
#             algos = st.multiselect("Compare Algos", ["avoa","co","coa","do","fla","gto","hba","hoa","pfo","poa","rime","roa","rsa","sto"], default=st.session_state.algos)
#             kpi_cmp = st.selectbox("KPI to Compare", ["fitness","average_sinr","fairness"], index=["fitness","average_sinr","fairness"].index(st.session_state.kpi_cmp))
#         if mode in ["MARL", "Hybrid"]:
#             marl_steps = st.slider("MARL Steps/Epoch", 1, 50, st.session_state.marl_steps)
#             viz_freq = st.slider("Visualization Frequency", 1, 20, st.session_state.viz_freq,
#                                  help="Update topology every N steps")
#         run = st.button("Start")
#         if run:
#             st.session_state.run_active = True
#             st.session_state.mode = mode
#             st.session_state.num_bs = num_bs
#             st.session_state.num_ue = num_ue
#             if mode in ["Single Metaheuristic", "Hybrid"]:
#                 st.session_state.meta_algo = meta_algo
#             if mode == "Comparison":
#                 st.session_state.algos = algos
#                 st.session_state.kpi_cmp = kpi_cmp
#             if mode in ["MARL", "Hybrid"]:
#                 st.session_state.marl_steps = marl_steps
#                 st.session_state.viz_freq = viz_freq
#             st.rerun()
#     else:
#         st.write(f"Running mode: **{st.session_state.mode}**")
#         if st.button("Reset Dashboard"):
#             for key in ["run_active","thread_active","thread_start","current_solution",
#                         "metrics","meta_result","comparison_results","comparison_trackers",
#                         "comparison_envs","best_algorithm","step_counter"]:
#                 st.session_state[key] = None if key in ["current_solution","meta_result"] else False if key.endswith("active") else {} if key in ["metrics"] else [] if key in ["comparison_results","comparison_trackers","comparison_envs"] else 0
#             st.session_state.tracker = KPITracker()
#             st.experimental_rerun()

#     # --- DEBUG PANEL ---
#     with st.expander("🔧 DEBUG STATE", expanded=False):
#         st.write({
#             "run_active":       st.session_state.run_active,
#             "thread_start":     st.session_state.thread_start,
#             "thread_active":    st.session_state.thread_active,
#             "step_counter":     st.session_state.step_counter,
#             "current_solution": (st.session_state.current_solution[:5] if st.session_state.current_solution else None),
#             "metrics":          st.session_state.metrics,
#         })

# # flicker-free plot helper
# def clear_and_plot(ph, fig, key):
#     # figs = st.session_state.figures
#     figs = st.session_state["figures"]  
#     if key in figs and figs[key] is not None:
#         st.session_state[key] += 1
#         figs[key] = copy.deepcopy(fig)
#         ph.plotly_chart(fig, use_container_width=True, key=f"{key}_{st.session_state[key]}")
#     else:
#         ph.empty()
#         st.session_state[key] += 1
#         ph.plotly_chart(fig, use_container_width=True, key=f"{key}_{st.session_state[key]}")
#         figs[key] = fig

# # topology drawer
# def render_topology(env, solution=None):
#     bs_coords = np.array([b.position for b in env.base_stations])
#     ue_coords = np.array([u.position for u in env.ues])
    
#     fig = go.Figure()
    
#     # BS markers + icons
#     fig.add_trace(go.Scatter(
#         x=bs_coords[:, 0], 
#         y=bs_coords[:, 1], 
#         mode='markers', 
#         name='BS', 
#         marker=dict(size=0, color='rgba(0,0,0,0)'), 
#         showlegend=True
#     ))
    
#     for x, y in bs_coords:
#         fig.add_layout_image(
#             source=f"data:image/png;base64,{bs_b64}", 
#             xref="x", 
#             yref="y", 
#             x=x, 
#             y=y,
#             sizex=5, 
#             sizey=5, 
#             xanchor="center", 
#             yanchor="middle", 
#             layer="above"
#         )
    
#     # If solution is provided, draw connection lines
#     if solution is not None and len(solution) > 0:
#         for i, (ux, uy) in enumerate(ue_coords):
#             bs_id = solution[i]
#             bx, by = bs_coords[bs_id]
#             fig.add_trace(go.Scatter(
#                 x=[bx, ux], 
#                 y=[by, uy], 
#                 mode='lines', 
#                 line=dict(color='lightgray', width=1), 
#                 showlegend=False
#             ))
#         # Add UEs with connection info
#         custom = np.stack([np.arange(len(ue_coords)), solution], axis=1)
#     else:
#         # Add UEs without connection info
#         custom = np.stack([np.arange(len(ue_coords)), [-1]*len(ue_coords)], axis=1)
    
#     # Add UE markers with customdata
#     fig.add_trace(go.Scatter(
#         x=ue_coords[:, 0], 
#         y=ue_coords[:, 1], 
#         mode='markers', 
#         name='UE', 
#         marker=dict(size=10),
#         hovertemplate="UE %{customdata[0]}<br>Assigned BS %{customdata[1]}<extra></extra>",
#         customdata=custom
#     ))
    
#     fig.update_layout(
#         title="Network Topology", 
#         xaxis=dict(scaleanchor="y"), 
#         yaxis=dict(constrain="domain")
#     )
    
#     return fig

# # layout containers
# col1, col2 = st.columns([3, 1])
# with col1:
#     ph_topo = st.expander("Network Topology", expanded=True).empty()
# with col2:
#     st.markdown("### Algorithm Info")
#     if st.session_state.run_active:
#         mode = st.session_state.mode
#         if mode in ["Single Metaheuristic", "Hybrid"]:
#             meta_algo = st.session_state.meta_algo
#             info = algo_info.get(meta_algo, {})
#             st.write(f"**{info.get('name', meta_algo)}**")
#             st.write(info.get("short", ""))
#         elif mode == "MARL":
#             info = algo_info.get("marl", {})
#             st.write("**MARL (PPO)**")
#             st.write(info.get("short", ""))
#         elif mode == "Comparison":
#             algos = st.session_state.algos
#             for a in algos:
#                 info = algo_info.get(a, {})
#                 st.write(f"**{info.get('name', a)}**: {info.get('short', '')}")
                
# # status / KPI / live containers
# ph_status=st.empty(); ph_progress=st.empty(); ph_kpi=st.empty(); ph_live=st.empty()

# # init env
# if not st.session_state.run_active or 'env' not in st.session_state:
#     st.session_state.env = NetworkEnvironment({"num_bs":st.session_state.num_bs,"num_ue":st.session_state.num_ue,"episode_length":100}, log_kpis=False)
# env = st.session_state.env

# # always show initial or current topology
# topo_fig = render_topology(env, st.session_state.current_solution)
# clear_and_plot(ph_topo, topo_fig, "topo_count")
# # Create status containers for various information
# # ph_status = st.empty()
# ph_progress = st.empty()
# ph_kpi = st.empty()
# ph_live_chart = st.empty()

# # # Initialize environment or use existing one
# # if "env" not in st.session_state or not st.session_state.run_active:
# #     env = NetworkEnvironment({"num_bs": st.session_state.get("num_bs", 10), 
# #                              "num_ue": st.session_state.get("num_ue", 50), 
# #                              "episode_length": 100}, log_kpis=False)
# #     st.session_state.env = env
# # else:
# #     env = st.session_state.env

# # # Initial topology
# # if not st.session_state.run_active:
# #     initial_topo = render_topology(env)
# #     ph_topo.plotly_chart(initial_topo, use_container_width=True)
# # Always draw something before threads start producing solutions

# # if st.session_state.current_solution is None:
# #     topo_fig = render_topology(env, None)
# #     clear_and_plot(ph_topo, topo_fig, "topo_count")


# # Background thread functions for different modes
# def run_meta_thread():
#     # tracker = st.session_state.tracker
#         # Safely grab (or create) our KPITracker inside the thread
#     tracker = st.session_state.get("tracker")
#     if tracker is None:
#         tracker = KPITracker()
#         st.session_state.tracker = tracker

#     # meta_algo = st.session_state.meta_algo
#     meta_algo = st.session_state.get("meta_algo") or "pfo"
#     env = st.session_state.env
    
#     st.session_state.thread_active = True
#     result = run_metaheuristic(env, meta_algo, epoch=0, kpi_logger=tracker, visualize_callback=None)
#     st.session_state.meta_result = result
#     st.session_state.current_solution = result['solution']
#     st.session_state.metrics = result['metrics']
#     st.session_state.thread_active = False

# def run_comparison_thread():
#     algos = st.session_state.algos
#     kpi_cmp = st.session_state.kpi_cmp
#     num_bs = st.session_state.num_bs
#     num_ue = st.session_state.num_ue
    
#     trackers = {}
#     results = {}
#     env_dict = {}
    
#     st.session_state.thread_active = True
    
#     for a in algos:
#         tr = KPITracker()
#         trackers[a] = tr
#         e = NetworkEnvironment({"num_bs": num_bs, "num_ue": num_ue}, log_kpis=False)
#         env_dict[a] = e
#         results[a] = run_metaheuristic(e, a, 0, tr, None)
    
#     # Store results in session state
#     st.session_state.comparison_results = results
#     st.session_state.comparison_trackers = trackers
#     st.session_state.comparison_envs = env_dict
    
#     # Find best algorithm
#     best_algo = None
#     best_value = float('-inf') if kpi_cmp != 'fairness' else 0
    
#     for algo, result in results.items():
#         if algo in trackers and not trackers[algo].history.empty:
#             final_value = trackers[algo].history[kpi_cmp].iloc[-1]
#             if final_value > best_value:
#                 best_value = final_value
#                 best_algo = algo
    
#     st.session_state.best_algorithm = best_algo
#     st.session_state.thread_active = False

# def run_marl_thread(initial_solution=None):
#     env_cfg = {"num_bs": st.session_state.num_bs, 
#                "num_ue": st.session_state.num_ue, 
#                "episode_length": 100}
#     ray_res = {"num_cpus": 4, "num_gpus": 0}
#     marl_steps = st.session_state.marl_steps
#     total_epochs = st.session_state.total_epochs
#     tracker = st.session_state.tracker
    
#     st.session_state.thread_active = True
#     st.session_state.step_counter = 0
    
#     for metrics, sol in run_marl(env_cfg, ray_res, initial_solution, marl_steps, total_epochs=total_epochs):
#         st.session_state.step_counter += 1
#         st.session_state.current_solution = sol
#         st.session_state.metrics = metrics
#         # Small sleep to give the main thread a chance to update
#         time.sleep(0.1)
    
#     st.session_state.thread_active = False

# def run_hybrid_thread():
#     tracker = st.session_state.tracker
#     meta_algo = st.session_state.meta_algo
#     env = st.session_state.env
    
#     # First run metaheuristic
#     st.session_state.hybrid_phase = "meta"
#     st.session_state.thread_active = True
    
#     meta_result = run_metaheuristic(env, meta_algo, 0, tracker, None)
#     st.session_state.meta_result = meta_result
#     st.session_state.current_solution = meta_result['solution']
#     st.session_state.metrics = meta_result['metrics']
    
#     # Now run MARL with the metaheuristic solution
#     st.session_state.hybrid_phase = "marl"
#     time.sleep(1)  # Give UI time to update phase
    
#     run_marl_thread(meta_result['solution'])

# # --- Main logic flow based on current state ---
# if st.session_state.run_active and not st.session_state.thread_start:
#     st.session_state.thread_start = True
#     mode = st.session_state.mode
#     if mode=="Single Metaheuristic": threading.Thread(target=run_meta_thread,daemon=True).start()
#     elif mode=="Comparison": threading.Thread(target=run_comparison_thread,daemon=True).start()
#     elif mode=="MARL": threading.Thread(target=run_marl_thread,daemon=True).start()
#     elif mode=="Hybrid": threading.Thread(target=run_hybrid_thread,daemon=True).start()
    
#     # Always display current solution if available (moved outside thread_active check)
#     if st.session_state.current_solution is not None:
#         current_topo = render_topology(env, st.session_state.current_solution)
#         clear_and_plot(ph_topo, current_topo, "topo_count")
    
#     # Always display progress bar for MARL-based modes (moved outside thread_active check)
#     if mode in ["MARL", "Hybrid"] and st.session_state.step_counter > 0:
#         progress_value = min(st.session_state.step_counter / st.session_state.total_epochs, 1.0)
#         ph_progress.progress(progress_value)
        
#         # Display phase indicator for hybrid mode
#         if mode == "Hybrid":
#             phase = st.session_state.get("hybrid_phase", "")
#             if phase == "meta":
#                 ph_status.info("Running metaheuristic warm-start...")
#             elif phase == "marl":
#                 ph_status.info(f"MARL refinement phase: Step {st.session_state.step_counter}/{st.session_state.total_epochs}")
    
#     # Always display KPI charts based on mode (moved outside thread_active check)
#     tracker = st.session_state.tracker
#     h = tracker.history
    
#     if not h.empty:
#         if mode == "Single Metaheuristic":
#             fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["Fitness", "SINR", "Fairness"])
#             fig.update_layout(height=600, margin=dict(t=50, b=40, l=40, r=40))
#             for i in (1, 2, 3): 
#                 fig.update_yaxes(showgrid=True, gridwidth=1, row=i, col=1)
#                 fig.update_xaxes(showgrid=True, gridwidth=1, row=i, col=1)
#             fig.add_trace(go.Scatter(x=h.index, y=h.fitness, name="Fitness", line=dict(width=2)), row=1, col=1)
#             fig.add_trace(go.Scatter(x=h.index, y=h.average_sinr, name="SINR", line=dict(width=2)), row=2, col=1)
#             fig.add_trace(go.Scatter(x=h.index, y=h.fairness, name="Fairness", line=dict(width=2)), row=3, col=1)
#             clear_and_plot(ph_kpi, fig, "kpi_count")
            
#         elif mode in ["MARL", "Hybrid"]:
#             # Display different metrics for MARL modes
#             if "reward_mean" in h:
#                 fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["Mean Reward", "Min Reward", "Max Reward"])
#                 fig.update_layout(height=600, margin=dict(t=50, b=40, l=40, r=40))
#                 for i in (1, 2, 3): 
#                     fig.update_yaxes(showgrid=True, gridwidth=1, row=i, col=1)
#                     fig.update_xaxes(showgrid=True, gridwidth=1, row=i, col=1)
#                 fig.add_trace(go.Scatter(x=h.index, y=h.reward_mean, name="Mean", line=dict(width=2)), row=1, col=1)
#                 fig.add_trace(go.Scatter(x=h.index, y=h.reward_min, name="Min", line=dict(width=2)), row=2, col=1)
#                 fig.add_trace(go.Scatter(x=h.index, y=h.reward_max, name="Max", line=dict(width=2)), row=3, col=1)
#                 clear_and_plot(ph_kpi, fig, "kpi_count")
    
#     # Always display live comparison chart (moved outside thread_active check)
#     if mode == "Comparison" and "comparison_trackers" in st.session_state:
#         trackers = st.session_state.comparison_trackers
#         kpi_cmp = st.session_state.kpi_cmp
        
#         fig_live = make_subplots(rows=1, cols=1)
#         for a, tr in trackers.items():
#             h = tr.history
#             if not h.empty and kpi_cmp in h:
#                 fig_live.add_trace(go.Scatter(x=h.index, y=h[kpi_cmp], name=a))
#         clear_and_plot(ph_live_chart, fig_live, "live_count")
        
#         # Show final comparison results if available
#         if "comparison_results" in st.session_state and not st.session_state.thread_active:
#             results = st.session_state.comparison_results
#             c1, _, c3 = st.columns([4, 1, 3])
#             with c1:
#                 st.subheader(f"Final {kpi_cmp.replace('_', ' ').title()}")
#                 df = pd.DataFrame([{"alg": a, kpi_cmp: r["metrics"][kpi_cmp]} for a, r in results.items()]).set_index("alg")
#                 st.plotly_chart(go.Figure(data=[go.Bar(x=df.index, y=df[kpi_cmp])]), use_container_width=True)
            
#             with c3:
#                 st.subheader("Final KPI Summary")
#                 for a, r in results.items():
#                     st.markdown(f"### {a.upper()}")
#                     st.write(f"{r['metrics'][kpi_cmp]:.3f}")
            
#             # Show best algorithm topology
#             best_algo = st.session_state.best_algorithm
#             if best_algo and "comparison_envs" in st.session_state:
#                 env_dict = st.session_state.comparison_envs
#                 best_value = results[best_algo]["metrics"][kpi_cmp]
#                 st.write(f"Showing topology for best algorithm: **{best_algo}** (best {kpi_cmp}: {best_value:.4f})")
#                 best_topo = render_topology(env_dict[best_algo], results[best_algo]['solution'])
#                 clear_and_plot(ph_topo, best_topo, "topo_count")
    
#     # Display final KPIs for metaheuristic if done
#     mr = st.session_state.get("meta_result")
#     if mode == "Single Metaheuristic" and mr and not st.session_state.thread_active:
#         metrics = mr["metrics"]
#         st.markdown("### Final KPIs Summary")
#         kpi_labels = {
#             "fitness": ("Fitness Score", ""),
#             "average_sinr": ("Avg SINR", "dB"),
#             "fairness": ("Fairness Index", "")
#         }
#         cols = st.columns(3)
#         for i, (kpi, (label, unit)) in enumerate(kpi_labels.items()):
#             val = metrics.get(kpi, 0)
#             cols[i].metric(label, f"{val:.3f} {unit}")
#         ph_status.success("Metaheuristic done")
    
#     # Start threads if needed
#     if not "thread_start" in st.session_state:
#         st.session_state.thread_start = True
        
#         if mode == "Single Metaheuristic":
#             thread = threading.Thread(target=run_meta_thread, daemon=True)
#             thread.start()
#         elif mode == "Comparison":
#             thread = threading.Thread(target=run_comparison_thread, daemon=True)
#             thread.start()
#         elif mode == "MARL":
#             thread = threading.Thread(target=run_marl_thread, daemon=True)
#             thread.start()
#         elif mode == "Hybrid":
#             thread = threading.Thread(target=run_hybrid_thread, daemon=True)
#             thread.start()
            
#         # Force a rerun to start displaying initial state
#         time.sleep(0.1)  # Small delay to ensure thread has started
#         st.rerun()
    
    # # Refresh page while thread is active
    # if st.session_state.thread_active:
    #     time.sleep(0.5)  # Short delay
    #     st.rerun()

# import asyncio 
# # Ensure there's an active event loop for Streamlit
# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())

# import streamlit as st
# import sys, os, threading, time, base64
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import copy
# import ray
# from collections import OrderedDict
# import logging
# import gymnasium as gym
# import torch
# import torch.nn as nn
# import plotly.express as px
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# from ray.rllib.models import ModelCatalog
# from ray.rllib.algorithms.ppo import PPOConfig
# from typing import Dict
# import json
# # --- Load Base Station icon (before any plotting) ---
# icon_path = os.path.join(os.path.dirname(__file__), "assets", "bs_icon.png")
# with open(icon_path, "rb") as f:
#     bs_b64 = base64.b64encode(f.read()).decode()
    
# # --- Load algorithm info ---
# info_path = os.path.join(os.path.dirname(__file__), "assets", "algo_info.json")
# with open(info_path) as f:
#     algo_info = json.load(f)

# # Add MARL info to algo_info if not present
# if "marl" not in algo_info:
#     algo_info["marl"] = {
#         "name": "Multi-Agent Reinforcement Learning",
#         "short": "Deep RL approach for distributed user association",
#         "long": """
#         Multi-Agent Reinforcement Learning (MARL) is a machine learning approach where multiple autonomous agents 
#         learn to interact with an environment and each other. In this 6G network optimization context, each UE is an 
#         agent making its own decisions about which BS to connect to, based on local observations.
        
#         Key features:
#         - Distributed decision-making: Each UE decides independently
#         - Adaptive learning: Agents adjust to changing network conditions
#         - Coordination: Implicit coordination emerges through shared environment
#         - Scalability: Parameter sharing allows handling of large networks
#         """,
#         "image": "assets/bs_icon.png"  # You may need to create this image
#     }

# # --- Page config ---
# st.set_page_config(page_title="6G Metaheuristic & MARL Dashboard", layout="wide")

# # --- Session-state initialization ---
# st.session_state.setdefault("cmp", {})
# for cnt in ("kpi_count", "live_count", "final_count", "topo_single_count", "marl_topo_count", "marl_kpi_count", "hybrid_topo_count"):  
#     st.session_state.setdefault(cnt, 0)
# if "figures" not in st.session_state:
#     st.session_state.figures = {
#         "topo": None,
#         "kpi": None,
#         "live": None,
#         "marl_topo": None,
#         "marl_kpi": None
#     }

# # Project path setup
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# sys.path.insert(0, project_root)
# from core.envs.custom_channel_env import NetworkEnvironment
# from core.hybrid_trainer.metaheuristic_opt import run_metaheuristic
# from core.hybrid_trainer.kpi_logger import KPITracker

# # --- Import the MetaPolicy model ---
# # This assumes your MetaPolicy class is in a module within the project structure
# # If it's not, you'll need to ensure it's imported correctly
# # from core.models.meta_policy import MetaPolicy

# # Define MetaPolicy here to ensure it's available
# class MetaPolicy(TorchModelV2, nn.Module):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
#         nn.Module.__init__(self)
#         TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        
#         # Extract config parameters
#         custom_config = model_config.get("custom_model_config", {})
#         self.initial_weights = custom_config.get("initial_weights", [])
#         self.num_bs = custom_config.get("num_bs", 5)
#         self.num_ue = custom_config.get("num_ue", 20)
        
#         # Calculate input size - one UE's observation size
#         if isinstance(obs_space, gym.spaces.Dict):
#             # For Dict space, we need the size of a single agent's observation
#             # This should be 2*num_bs+1
#             input_size = 2 * self.num_bs + 1
#         else:
#             input_size = np.prod(obs_space.shape)
            
#         print(f"Calculated input size: {input_size}")
        
#         # Enhanced network architecture for better performance
#         hidden_size = 64  # Add a hidden layer for more expressive policy
        
#         # Policy network for actions - each UE chooses from num_bs actions
#         self.policy_network = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, self.num_bs)  # Output should match number of BS options
#         )
        
#         # Value network (critic) - also with a hidden layer
#         self.value_network = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, 1)
#         )
        
#         # Store current value function output
#         self._cur_value = None
        
#         # Initialize weights using metaheuristic solution if available
#         if self.initial_weights:
#             self._apply_initial_weights()
            
#     def _apply_initial_weights(self):
#         """Apply initial weights to bias the policy"""
#         # Each UE should have its own policy preferences
#         with torch.no_grad():
#             # Get the last layer of the policy network
#             policy_output_layer = self.policy_network[-1]
            
#             # Initialize with small random weights for exploration
#             policy_output_layer.weight.data.normal_(0.0, 0.01)
#             policy_output_layer.bias.data.fill_(0.0)
            
#             # If we have initial weights from metaheuristic
#             if isinstance(self.initial_weights, list) and len(self.initial_weights) > 0:
#                 # Determine which UE this policy is for based on available context
#                 # In MARL with parameter sharing, we can't know for sure,
#                 # so we use a more general approach
                
#                 # Count the frequency of each BS in the solution
#                 bs_counts = np.zeros(self.num_bs)
#                 for bs_idx in self.initial_weights:
#                     if 0 <= bs_idx < self.num_bs:
#                         bs_counts[bs_idx] += 1
                
#                 # Bias toward less congested BSs
#                 total_ues = sum(bs_counts)
#                 if total_ues > 0:
#                     for bs_idx in range(self.num_bs):
#                         # Lower allocation ratio = higher bias
#                         congestion_factor = 1.0 - (bs_counts[bs_idx] / total_ues)
#                         policy_output_layer.bias.data[bs_idx] = congestion_factor * 1.0 # Stronger bias toward less congested BSs
                
#                 print(f"Applied metaheuristic bias based on BS congestion")
                
#     def forward(self, input_dict, state, seq_lens):
#         # Get observation from input dict
#         obs = input_dict["obs"]
        
#         # Debug: Check observation shape and values
#         # print(f"Forward input shape: {obs.shape if hasattr(obs, 'shape') else 'dict'}")
        
#         # Handle different input types
#         if isinstance(obs, dict) or isinstance(obs, OrderedDict):
#             # In MARL, each agent should only receive its own observation
#             # Convert all values to tensors and flatten if needed
#             x = torch.cat([torch.tensor(v).flatten() for v in obs.values()])
#         else:
#             # Already a tensor
#             x = obs.float() if isinstance(obs, torch.Tensor) else torch.FloatTensor(obs)
            
#         # Ensure batch dimension
#         if x.dim() == 1:
#             x = x.unsqueeze(0)
        
#         # Forward passes
#         logits = self.policy_network(x)
#         self._cur_value = self.value_network(x).squeeze(-1)
        
#         return logits, state

#     def value_function(self):
#         """Return value function output for current observation"""
#         # This is required for PPO training
#         assert self._cur_value is not None, "value function not calculated"
#         return self._cur_value

# # Register the custom model
# ModelCatalog.register_custom_model("meta_policy", MetaPolicy)

# # --- HybridTraining class ---
# class HybridTraining:
#     def __init__(self, config: Dict):
#         # Initialize Ray if not already initialized
#         if not ray.is_initialized():
#             ray.init(
#                 runtime_env={
#                     "env_vars": {"PYTHONPATH": project_root},
#                     "working_dir": project_root
#                 },
#                 logging_level=logging.INFO,
#                 log_to_driver=True,
#                 **config.get("ray_resources", {})
#             )
        
#         self.config = config
#         self.env = NetworkEnvironment(config["env_config"])
#         self.obs_space = self.env.observation_space
#         self.act_space = self.env.action_space
#         self.kpi_logger = KPITracker(enabled=config.get("logging", {}).get("enabled", True))
#         self.current_epoch = 0
#         self.metaheuristic_runs = 0
#         self.max_metaheuristic_runs = 1
#         self.visualize_callback = config.get("visualize_callback", None)
        
#         # Create log directory if needed
#         if config.get("logging", {}).get("enabled", False):
#             os.makedirs(config.get("logging", {}).get("log_dir", "./logs"), exist_ok=True)
            
#     def _execute_marl_phase(self, initial_policy: Dict=None):
#         """Execute MARL training phase with visualization callback"""
#         print(f"\nStarting {self.config.get('marl_algorithm', 'PPO').upper()} training...")
        
#         # Extract initial solution safely
#         initial_weights = []
#         if initial_policy is not None:
#             initial_weights = initial_policy.tolist() if hasattr(initial_policy, 'tolist') else list(initial_policy)
#             assert len(initial_weights) == self.config["env_config"]["num_ue"]
        
#         env_config = {
#             **self.config["env_config"]
#         }
        
#         # MARL configuration
#         marl_config = (
#             PPOConfig()
#             .environment(
#                 "NetworkEnv",
#                 env_config=env_config
#             )
#             .training(
#                 model={
#                     "custom_model": "meta_policy",
#                     "custom_model_config": {
#                         "initial_weights": initial_weights,
#                         "num_bs": self.config["env_config"]["num_bs"],
#                         "num_ue": self.config["env_config"]["num_ue"]
#                     }
#                 },
#                 gamma=0.99,
#                 lr=0.0005,
#                 lr_schedule=[(0, 0.00005), (1000, 0.0001), (10000, 0.0005)],
#                 entropy_coeff=0.01,
#                 kl_coeff=0.2,
#                 train_batch_size=4000,
#                 sgd_minibatch_size=128,
#                 num_sgd_iter=10,
#                 clip_param=0.2,
#             )
#             .env_runners(
#                 sample_timeout_s=3600,
#                 rollout_fragment_length=25
#             )
#             .multi_agent(
#                 policies={
#                     f"ue_{i}": (None, self.obs_space[f"ue_{i}"], self.act_space[f"ue_{i}"], {})
#                     for i in range(self.config["env_config"]["num_ue"])
#                 },
#                 policy_mapping_fn=lambda agent_id, episode=None, worker=None, **kwargs: agent_id
#             )
#         )
        
#         analysis = ray.tune.run(
#             "PPO",
#             config=marl_config.to_dict(),
#             stop={"training_iteration": self.config.get("marl_steps_per_phase", 10)},
#             checkpoint_at_end=True,
#             callbacks=[self._create_marl_callback()]
#         )
#         print("Trial ended with status:", analysis.trials[0].status)
        
#         # Get the best checkpoint
#         best_trial = analysis.get_best_trial("env_runners/episode_return_mean", mode="max")
#         if best_trial:
#             best_checkpoint = analysis.get_best_checkpoint(best_trial, "env_runners/episode_return_mean", mode="max")
#             print(f"Best checkpoint: {best_checkpoint}")
            
#             # Extract the final policy
#             if best_checkpoint:
#                 # TODO: Logic to extract trained policy for visualization
#                 pass
        
#         return analysis
    
#     def _create_marl_callback(self):
#         """Create a Ray Tune callback for tracking training progress"""
        
#         class MARLCallback(ray.tune.Callback):
#             def __init__(self, parent):
#                 self.parent = parent
                
#             def on_trial_result(self, iteration, trials, trial, result, **info):
#                 # Get reward values from result dictionary
#                 reward_mean = result.get("env_runners/episode_return_mean", 0)
#                 reward_min = result.get("env_runners/episode_return_min", 0)
#                 reward_max = result.get("env_runners/episode_return_max", 0)
#                 episode_len = result.get("env_runners/episode_len_mean", 0)
                
#                 # Extract current policy's decisions (user associations)
#                 # This requires additional logic to get the associations from the training
#                 # Let's use a placeholder for now
#                 current_associations = None
                
#                 # Try to extract user associations if available
#                 env_info = result.get("custom_metrics", {}).get("env_info", None)
#                 if env_info and "user_associations" in env_info:
#                     current_associations = env_info["user_associations"]
                
#                 print(f"[Trial {trial.trial_id}] Iter {result['training_iteration']} | "
#                       f"Reward: {reward_mean:.3f} | Length: {episode_len:.1f}")
                
#                 if self.parent.kpi_logger and self.parent.kpi_logger.enabled:
#                     metrics = {
#                         "iteration": self.parent.current_epoch * self.parent.config.get("marl_steps_per_phase", 10) + 
#                                 result["training_iteration"],
#                         "reward_mean": reward_mean,
#                         "reward_min": reward_min, 
#                         "reward_max": reward_max,
#                         "episode_length": episode_len
#                     }
                    
#                     # Add SINR and fairness metrics if available
#                     if "average_sinr" in result.get("custom_metrics", {}):
#                         metrics["average_sinr"] = result["custom_metrics"]["average_sinr"]
#                     if "fairness" in result.get("custom_metrics", {}):
#                         metrics["fairness"] = result["custom_metrics"]["fairness"]
                    
#                     self.parent.kpi_logger.log_metrics(
#                         phase="marl",
#                         algorithm="PPO",
#                         metrics=metrics,
#                         episode=result.get("training_iteration", 0)
#                     )
                
#                 # Call visualization callback if provided
#                 if self.parent.visualize_callback and current_associations is not None:
#                     self.parent.visualize_callback(metrics, current_associations)
                
#         return MARLCallback(self)
        
#     def run_hybrid_training(self, initial_metaheuristic=None):
#         """Run a complete hybrid training cycle with optional initial metaheuristic"""
        
#         # Start with metaheuristic optimization if requested
#         metaheuristic_solution = None
#         if initial_metaheuristic:
#             print(f"Running initial metaheuristic optimization with {initial_metaheuristic}...")
#             meta_result = run_metaheuristic(
#                 env=self.env,
#                 algorithm=initial_metaheuristic,
#                 epoch=0,
#                 kpi_logger=self.kpi_logger,
#                 visualize_callback=self.visualize_callback
#             )
#             metaheuristic_solution = meta_result.get("solution")
#             print(f"Metaheuristic optimization complete. Solution: {metaheuristic_solution}")
        
#         # Execute MARL training with the initial policy from metaheuristic
#         print("Starting MARL training phase...")
#         marl_result = self._execute_marl_phase(initial_policy=metaheuristic_solution)
        
#         return {
#             "metaheuristic_result": meta_result if initial_metaheuristic else None,
#             "marl_result": marl_result,
#             "final_metrics": self.kpi_logger.history.iloc[-1].to_dict() if not self.kpi_logger.history.empty else {}
#         }

# st.title("6G Metaheuristic & MARL Dashboard")

# # Helper: clear & plot
# def clear_and_plot(ph, fig, counter_name, force_redraw=False):
#     # Only clears and redraws the given placeholder
#     if not force_redraw and counter_name in st.session_state.figures and st.session_state.figures[counter_name] is not None:
#         st.session_state[counter_name] += 1
#         st.session_state.figures[counter_name] = copy.deepcopy(fig)
#         ph.plotly_chart(fig, use_container_width=True, key=f"{counter_name}_{st.session_state[counter_name]}")
#     else:
#         ph.empty()
#         st.session_state[counter_name] += 1
#         ph.plotly_chart(fig, use_container_width=True, key=f"{counter_name}_{st.session_state[counter_name]}")
#         st.session_state.figures[counter_name] = copy.deepcopy(fig)

# # Sidebar controls
# with st.sidebar:
#     mode = st.radio("Mode", ["Single Metaheuristic", "Comparison", "MARL", "Hybrid"])
#     num_bs = st.slider("Base Stations", 5, 50, 10)
#     num_ue = st.slider("Users", 20, 200, 50)
    
#     if mode in ["Single Metaheuristic", "Hybrid"]:
#         metaheuristic_algorithm = st.selectbox(
#             "Metaheuristic Algorithm", 
#             ["pfo", "co", "coa", "do", "fla", "gto", "hba", "hoa", "avoa", "poa", "rime", "roa", "rsa", "sto"]
#         )
    
#     if mode == "Comparison":
#         algorithms = st.multiselect(
#             "Compare Algorithms", 
#             ["avoa", "co", "coa", "do", "fla", "gto", "hba", "hoa", "pfo", "poa", "rime", "roa", "rsa", "sto"], 
#             default=["pfo", "co"]
#         )
#         kpi_to_compare = st.selectbox("KPI to Compare", ["fitness", "average_sinr", "fairness"], index=0)
    
#     if mode in ["MARL", "Hybrid"]:
#         marl_steps = st.slider("MARL Training Steps", 5, 50, 10)
        
#     run = st.button("Start")

# # Top row: topology + info
# col_topo, col_info = st.columns([3, 1])
# with col_topo:
#     st.markdown("---")
#     ph_topo = st.expander("Network Topology", expanded=True).empty()
#     topo_env = NetworkEnvironment({"num_bs": num_bs, "num_ue": num_ue}, log_kpis=False)
#     bs_coords = np.array([b.position for b in topo_env.base_stations])
#     ue_coords = np.array([u.position for u in topo_env.ues])
    
#     # Initial topology plot
#     fig_topo = go.Figure()
#     fig_topo.add_trace(go.Scatter(
#         x=bs_coords[:, 0], 
#         y=bs_coords[:, 1], 
#         mode='markers', 
#         name='BS', 
#         marker=dict(size=0, color='rgba(0,0,0,0)'), 
#         showlegend=True
#     ))
    
#     # Add BS icons
#     for x, y in bs_coords:
#         fig_topo.add_layout_image(
#             source=f"data:image/png;base64,{bs_b64}", 
#             xref="x", 
#             yref="y", 
#             x=x, 
#             y=y,
#             sizex=5, 
#             sizey=5, 
#             xanchor="center", 
#             yanchor="middle", 
#             layer="above"
#         )
        
#     # Add UEs
#     fig_topo.add_trace(go.Scatter(
#         x=ue_coords[:, 0], 
#         y=ue_coords[:, 1], 
#         mode='markers', 
#         name='UE', 
#         marker=dict(size=10),
#         hovertemplate="UE %{customdata[0]}<br>Assigned BS %{customdata[1]}<extra></extra>",
#         customdata=np.stack([np.arange(len(ue_coords)), [-1]*len(ue_coords)], axis=1)
#     ))
    
#     ph_topo.plotly_chart(fig_topo, use_container_width=True)
#     st.markdown("---")
    
# with col_info:
#     with st.expander("Algorithm Info"):
#         if mode == "Single Metaheuristic":
#             info = algo_info.get(metaheuristic_algorithm, {})
#             st.markdown(f"## {info.get('name', metaheuristic_algorithm)}")
#             st.image(info.get("image"), use_container_width=True)
#             st.write(info.get("long", "No description available."))
#         elif mode == "MARL":
#             info = algo_info.get("marl", {})
#             st.markdown(f"## {info.get('name', 'MARL')}")
#             st.image(info.get("image"), use_container_width=True)
#             st.write(info.get("long", "No description available."))
#         elif mode == "Hybrid":
#             meta_info = algo_info.get(metaheuristic_algorithm, {})
#             marl_info = algo_info.get("marl", {})
#             st.markdown(f"## Hybrid: {meta_info.get('name', metaheuristic_algorithm)} + {marl_info.get('name', 'MARL')}")
#             st.write("This approach combines metaheuristic optimization for initial solution with MARL for dynamic adaptation.")
#             st.write(meta_info.get("short", ""))
#             st.write(marl_info.get("short", ""))
#         else:  # Comparison
#             for alg in algorithms:
#                 info = algo_info.get(alg, {})
#                 st.markdown(f"**{info.get('name', alg)}**: {info.get('short','')}")

# # SINGLE METAHEURISTIC MODE
# if run and mode == "Single Metaheuristic":
#     ph_kpi = st.empty()
#     tracker = KPITracker()
#     env = topo_env
    
#     def visualize(metrics, solution):
#         # Visualize KPI history
#         hist = tracker.history
#         if not hist.empty:
#             fig_kpi = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["Fitness", "SINR", "Fairness"])
#             fig_kpi.update_layout(height=600, margin=dict(t=50, b=40, l=40, r=40))
            
#             for i in (1, 2, 3): 
#                 fig_kpi.update_yaxes(showgrid=True, gridwidth=1, row=i, col=1)
#                 fig_kpi.update_xaxes(showgrid=True, gridwidth=1, row=i, col=1)
                
#             fig_kpi.add_trace(go.Scatter(x=hist.index, y=hist['fitness'], name='Fitness', line=dict(width=2)), row=1, col=1)
#             fig_kpi.add_trace(go.Scatter(x=hist.index, y=hist['average_sinr'], name='SINR', line=dict(width=2)), row=2, col=1)
#             fig_kpi.add_trace(go.Scatter(x=hist.index, y=hist['fairness'], name='Fairness', line=dict(width=2)), row=3, col=1)
            
#             clear_and_plot(ph_kpi, fig_kpi, "kpi_count")
        
#         # Visualize network topology with connections
#         bs_coords = np.array([b.position for b in env.base_stations])
#         ue_coords = np.array([u.position for u in env.ues])
        
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(
#             x=bs_coords[:, 0], 
#             y=bs_coords[:, 1], 
#             mode='markers', 
#             name='BS', 
#             marker=dict(size=0, color='rgba(0,0,0,0)'), 
#             showlegend=True
#         ))
        
#         # Add BS icons
#         for x, y in bs_coords: 
#             fig.add_layout_image(
#                 source=f"data:image/png;base64,{bs_b64}", 
#                 xref="x", 
#                 yref="y", 
#                 x=x, 
#                 y=y, 
#                 sizex=5, 
#                 sizey=5, 
#                 xanchor="center", 
#                 yanchor="middle", 
#                 layer="above"
#             )
        
#         # Add connection lines between UEs and BSs
#         for i, (ux, uy) in enumerate(ue_coords): 
#             if i < len(solution):  # Safety check
#                 bs_idx = solution[i]
#                 if 0 <= bs_idx < len(bs_coords):  # Safety check
#                     bx, by = bs_coords[bs_idx]
#                     fig.add_trace(go.Scatter(
#                         x=[bx, ux],
#                         y=[by, uy],
#                         mode='lines',
#                         line=dict(color='lightgray', width=1),
#                         showlegend=False
#                     ))
        
#         # Add UEs with custom data for hover
#         custom = np.stack([np.arange(len(ue_coords)), solution], axis=1)
#         fig.add_trace(go.Scatter(
#             x=ue_coords[:, 0],
#             y=ue_coords[:, 1],
#             mode='markers',
#             name='UE',
#             marker=dict(size=10),
#             hovertemplate="UE %{customdata[0]}<br>BS %{customdata[1]}<extra></extra>",
#             customdata=custom
#         ))
        
#         clear_and_plot(ph_topo, fig, "topo_single_count")
    
#     # Run metaheuristic optimization
#     res = run_metaheuristic(
#         env=env, 
#         algorithm=metaheuristic_algorithm, 
#         epoch=0, 
#         kpi_logger=tracker, 
#         visualize_callback=visualize
#     )
    
#     metrics = res["metrics"]
#     st.success("Single optimization complete!")
#     st.markdown("### Final KPIs Summary")

#     kpi_labels = {
#         "fitness": ("Fitness Score", ""),
#         "average_sinr": ("Avg SINR", "dB"),
#         "fairness": ("Fairness Index", "")
#     }
    
#     cols = st.columns(3)
#     for i, (kpi, (label, unit)) in enumerate(kpi_labels.items()):
#         val = metrics.get(kpi, 0)
#         cols[i].metric(label, f"{val:.3f} {unit}")

# # MARL MODE
# elif run and mode == "MARL":
#     ph_marl_kpi = st.empty()
#     ph_marl_topo = ph_topo  # Use the same topology placeholder
    
#     tracker = KPITracker()
#     env = topo_env
    
#     def visualize_marl(metrics, solution):
#         # Visualize KPI history
#         hist = tracker.history
#         if not hist.empty:
#             fig_kpi = make_subplots(rows=3, cols=1, shared_xaxes=True, 
#                                 subplot_titles=["Reward", "SINR", "Fairness"])
#             fig_kpi.update_layout(height=600, margin=dict(t=50, b=40, l=40, r=40))
            
#             for i in (1, 2, 3): 
#                 fig_kpi.update_yaxes(showgrid=True, gridwidth=1, row=i, col=1)
#                 fig_kpi.update_xaxes(showgrid=True, gridwidth=1, row=i, col=1)
            
#             # Plot available metrics
#             if 'reward_mean' in hist.columns:
#                 fig_kpi.add_trace(go.Scatter(x=hist.index, y=hist['reward_mean'], name='Reward', line=dict(width=2)), 
#                                 row=1, col=1)
#             if 'average_sinr' in hist.columns:
#                 fig_kpi.add_trace(go.Scatter(x=hist.index, y=hist['average_sinr'], name='SINR', line=dict(width=2)), 
#                                 row=2, col=1)
#             if 'fairness' in hist.columns:
#                 fig_kpi.add_trace(go.Scatter(x=hist.index, y=hist['fairness'], name='Fairness', line=dict(width=2)), 
#                                 row=3, col=1)
            
#             clear_and_plot(ph_marl_kpi, fig_kpi, "marl_kpi_count")
        
#         # Visualize network topology with connections
#         if solution is not None:
#             bs_coords = np.array([b.position for b in env.base_stations])
#             ue_coords = np.array([u.position for u in env.ues])
            
#             fig = go.Figure()
            
#             # Add base stations (as icons)
#             fig.add_trace(go.Scatter(
#                 x=bs_coords[:, 0], 
#                 y=bs_coords[:, 1], 
#                 mode='markers', 
#                 name='BS', 
#                 marker=dict(size=0, color='rgba(0,0,0,0)'), 
#                 showlegend=True
#             ))
            
#             # Add BS icons
#             for i, (x, y) in enumerate(bs_coords):
#                 fig.add_layout_image(
#                     source=f"data:image/png;base64,{bs_b64}", 
#                     xref="x", 
#                     yref="y", 
#                     x=x, 
#                     y=y, 
#                     sizex=5, 
#                     sizey=5, 
#                     xanchor="center", 
#                     yanchor="middle", 
#                     layer="above"
#                 )
                
#                 # Add BS load indicators
#                 bs = env.base_stations[i]
#                 if hasattr(bs, 'load') and hasattr(bs, 'capacity'):
#                     load_pct = bs.load / bs.capacity if bs.capacity > 0 else 0
#                     fig.add_trace(go.Scatter(
#                         x=[x],
#                         y=[y+3],  # Position above the BS icon
#                         mode='markers',
#                         marker=dict(
#                             size=12,
#                             color=f'rgba({int(255*load_pct)}, {int(255*(1-load_pct))}, 0, 0.7)',
#                             line=dict(width=1, color='black')
#                         ),
#                         text=[f"BS {i}: {load_pct*100:.1f}%"],
#                         hoverinfo="text",
#                         showlegend=False
#                     ))
            
#             # Add connection lines between UEs and BSs
#             for i, ue in enumerate(env.ues):
#                 if ue.associated_bs is not None:
#                     bs_idx = ue.associated_bs
#                     if 0 <= bs_idx < len(bs_coords):  # Safety check
#                         ux, uy = ue_coords[i]
#                         bx, by = bs_coords[bs_idx]
#                         fig.add_trace(go.Scatter(
#                             x=[bx, ux],
#                             y=[by, uy],
#                             mode='lines',
#                             line=dict(
#                                 color=f'rgba(100, 100, 255, {max(0.2, min(1.0, (ue.sinr+30)/60))})',
#                                 width=1
#                             ),
#                             showlegend=False
#                         ))
            
#             # Add UEs
#             custom_data = []
#             for i, ue in enumerate(env.ues):
#                 bs_id = ue.associated_bs if ue.associated_bs is not None else -1
#                 sinr = ue.sinr if ue.sinr != -np.inf else -30
#                 custom_data.append([i, bs_id, f"{sinr:.2f}"])
            
#             fig.add_trace(go.Scatter(
#                 x=ue_coords[:, 0],
#                 y=ue_coords[:, 1],
#                 mode='markers',
#                 name='UE',
#                 marker=dict(
#                     size=10,
#                     color=[
#                         'red' if ue.associated_bs is None else 
#                         f'rgba(0, {min(255, int(128 + 128*(ue.sinr+30)/60))}, 0, 0.8)' 
#                         for ue in env.ues
#                     ]
#                 ),
#                 hovertemplate="UE %{customdata[0]}<br>BS %{customdata[1]}<br>SINR %{customdata[2]}dB<extra></extra>",
#                 customdata=custom_data
#             ))
            
#             clear_and_plot(ph_marl_topo, fig, "marl_topo_count")

# # --- IMPLEMENTATION FOR COMPARISON MODE ---
# elif run and mode == "Comparison":
#     # Create placeholders for comparison visualizations
#     ph_comparison = st.empty()
    
#     # Dictionary to store results for each algorithm
#     comparison_results = {}
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     # Setup tracker and environment
#     tracker = KPITracker()
#     env = topo_env
    
#     # Run each selected algorithm
#     for i, algorithm in enumerate(algorithms):
#         status_text.text(f"Running {algorithm}... ({i+1}/{len(algorithms)})")
        
#         if algorithm == "marl":
#             # Run MARL for comparison
#             hybrid = HybridTraining({
#                 "env_config": {
#                     "num_bs": num_bs,
#                     "num_ue": num_ue,
#                     "log_kpis": True
#                 },
#                 "marl_algorithm": "PPO",
#                 "marl_steps_per_phase": 10,
#                 "logging": {"enabled": True},
#                 "ray_resources": {"num_cpus": 2}
#             })
            
#             # Execute MARL without initial metaheuristic
#             marl_result = hybrid._execute_marl_phase()
            
#             # Extract the relevant metrics
#             best_trial = marl_result.get_best_trial("env_runners/episode_return_mean", mode="max")
#             if best_trial:
#                 metrics = best_trial.last_result
#                 comparison_results["marl"] = {
#                     "fitness": metrics.get("env_runners/episode_return_mean", 0),
#                     "average_sinr": metrics.get("custom_metrics", {}).get("average_sinr", 0),
#                     "fairness": metrics.get("custom_metrics", {}).get("fairness", 0),
#                     "iterations": list(range(1, metrics.get("training_iteration", 10) + 1)),
#                     "history": {
#                         "fitness": [r.get("env_runners/episode_return_mean", 0) for r in best_trial.metric_analysis["env_runners/episode_return_mean"]["hist"]],
#                         "average_sinr": [r.get("custom_metrics", {}).get("average_sinr", 0) for r in best_trial.metric_analysis["env_runners/episode_return_mean"]["hist"]],
#                         "fairness": [r.get("custom_metrics", {}).get("fairness", 0) for r in best_trial.metric_analysis["env_runners/episode_return_mean"]["hist"]]
#                     }
#                 }
#         else:
#             # Run metaheuristic for comparison
#             result = run_metaheuristic(
#                 env=env,
#                 algorithm=algorithm,
#                 epoch=0,
#                 kpi_logger=tracker,
#                 visualize_callback=None  # Don't visualize individual runs
#             )
            
#             # Store the results
#             metrics = result["metrics"]
#             history = tracker.history.copy()
            
#             comparison_results[algorithm] = {
#                 "fitness": metrics.get("fitness", 0),
#                 "average_sinr": metrics.get("average_sinr", 0),
#                 "fairness": metrics.get("fairness", 0),
#                 "iterations": list(range(1, len(history) + 1)),
#                 "history": {
#                     "fitness": history["fitness"].tolist() if "fitness" in history else [],
#                     "average_sinr": history["average_sinr"].tolist() if "average_sinr" in history else [],
#                     "fairness": history["fairness"].tolist() if "fairness" in history else []
#                 }
#             }
            
#             # Reset the tracker for the next algorithm
#             tracker = KPITracker()
        
#         # Update progress
#         progress_bar.progress((i + 1) / len(algorithms))
    
#     # Create comparison visualization
#     status_text.text("Creating comparison visualizations...")
    
#     # Plot the comparison results
#     fig_comparison = make_subplots(rows=1, cols=1, subplot_titles=[f"{kpi_to_compare.replace('_', ' ').title()} Comparison"])
#     fig_comparison.update_layout(height=500, margin=dict(t=50, b=40, l=40, r=40))
    
#     # Add traces for each algorithm
#     colors = px.colors.qualitative.Plotly
#     for i, (alg, results) in enumerate(comparison_results.items()):
#         color = colors[i % len(colors)]
        
#         # Get the KPI history
#         kpi_history = results["history"].get(kpi_to_compare, [])
#         iterations = results["iterations"]
        
#         if kpi_history:
#             fig_comparison.add_trace(
#                 go.Scatter(
#                     x=iterations[:len(kpi_history)],
#                     y=kpi_history,
#                     name=algo_info.get(alg, {}).get("name", alg),
#                     line=dict(width=2, color=color)
#                 )
#             )
    
#     # Update axis labels
#     fig_comparison.update_xaxes(title="Iteration")
#     fig_comparison.update_yaxes(title=kpi_to_compare.replace("_", " ").title())
    
#     # Display the comparison plot
#     clear_and_plot(ph_comparison, fig_comparison, "cmp_count")
    
#     # Display final metrics
#     st.markdown("### Final KPI Comparison")
    
#     # Create a DataFrame for the final metrics
#     final_metrics = {
#         "Algorithm": [],
#         "Fitness": [],
#         "Average SINR (dB)": [],
#         "Fairness": []
#     }
    
#     for alg, results in comparison_results.items():
#         final_metrics["Algorithm"].append(algo_info.get(alg, {}).get("name", alg))
#         final_metrics["Fitness"].append(f"{results.get('fitness', 0):.3f}")
#         final_metrics["Average SINR (dB)"].append(f"{results.get('average_sinr', 0):.3f}")
#         final_metrics["Fairness"].append(f"{results.get('fairness', 0):.3f}")
    
#     # Display as a table
#     df_final = pd.DataFrame(final_metrics)
#     st.table(df_final)
    
#     status_text.text("Comparison complete!")

# # --- IMPLEMENTATION FOR MARL MODE ---
# elif run and mode == "MARL":
#     # Create placeholders for MARL visualizations
#     ph_marl_kpi = st.empty()
#     ph_marl_topo = ph_topo  # Use the same topology placeholder
    
#     # Setup tracker and environment
#     tracker = KPITracker()
#     env = topo_env
    
#     # Visualization callback for MARL
#     def visualize_marl(metrics, solution):
#         # Visualize KPI history
#         hist = tracker.history
#         if not hist.empty:
#             fig_kpi = make_subplots(rows=3, cols=1, shared_xaxes=True, 
#                                   subplot_titles=["Reward", "SINR", "Fairness"])
#             fig_kpi.update_layout(height=600, margin=dict(t=50, b=40, l=40, r=40))
            
#             for i in (1, 2, 3): 
#                 fig_kpi.update_yaxes(showgrid=True, gridwidth=1, row=i, col=1)
#                 fig_kpi.update_xaxes(showgrid=True, gridwidth=1, row=i, col=1)
            
#             # Plot available metrics
#             if 'reward_mean' in hist.columns:
#                 fig_kpi.add_trace(go.Scatter(x=hist.index, y=hist['reward_mean'], 
#                                            name='Reward', line=dict(width=2)), 
#                                 row=1, col=1)
#             if 'average_sinr' in hist.columns:
#                 fig_kpi.add_trace(go.Scatter(x=hist.index, y=hist['average_sinr'], 
#                                            name='SINR', line=dict(width=2)), 
#                                 row=2, col=1)
#             if 'fairness' in hist.columns:
#                 fig_kpi.add_trace(go.Scatter(x=hist.index, y=hist['fairness'], 
#                                            name='Fairness', line=dict(width=2)), 
#                                 row=3, col=1)
            
#             clear_and_plot(ph_marl_kpi, fig_kpi, "marl_kpi_count")
        
#         # Visualize network topology with connections
#         if solution is not None:
#             bs_coords = np.array([b.position for b in env.base_stations])
#             ue_coords = np.array([u.position for u in env.ues])
            
#             fig = go.Figure()
            
#             # Add base stations (as icons)
#             fig.add_trace(go.Scatter(
#                 x=bs_coords[:, 0], 
#                 y=bs_coords[:, 1], 
#                 mode='markers', 
#                 name='BS', 
#                 marker=dict(size=0, color='rgba(0,0,0,0)'), 
#                 showlegend=True
#             ))
            
#             # Add BS icons
#             for i, (x, y) in enumerate(bs_coords):
#                 fig.add_layout_image(
#                     source=f"data:image/png;base64,{bs_b64}", 
#                     xref="x", 
#                     yref="y", 
#                     x=x, 
#                     y=y, 
#                     sizex=5, 
#                     sizey=5, 
#                     xanchor="center", 
#                     yanchor="middle", 
#                     layer="above"
#                 )
                
#                 # Add BS load indicators
#                 bs = env.base_stations[i]
#                 if hasattr(bs, 'load') and hasattr(bs, 'capacity'):
#                     load_pct = bs.load / bs.capacity if bs.capacity > 0 else 0
#                     fig.add_trace(go.Scatter(
#                         x=[x],
#                         y=[y+3],  # Position above the BS icon
#                         mode='markers',
#                         marker=dict(
#                             size=12,
#                             color=f'rgba({int(255*load_pct)}, {int(255*(1-load_pct))}, 0, 0.7)',
#                             line=dict(width=1, color='black')
#                         ),
#                         text=[f"BS {i}: {load_pct*100:.1f}%"],
#                         hoverinfo="text",
#                         showlegend=False
#                     ))
            
#             # Add connection lines between UEs and BSs
#             for i, ue in enumerate(env.ues):
#                 if hasattr(ue, 'associated_bs') and ue.associated_bs is not None:
#                     bs_idx = ue.associated_bs
#                     if 0 <= bs_idx < len(bs_coords):  # Safety check
#                         ux, uy = ue_coords[i]
#                         bx, by = bs_coords[bs_idx]
#                         sinr = getattr(ue, 'sinr', 0)
#                         fig.add_trace(go.Scatter(
#                             x=[bx, ux],
#                             y=[by, uy],
#                             mode='lines',
#                             line=dict(
#                                 color=f'rgba(100, 100, 255, {max(0.2, min(1.0, (sinr+30)/60))})',
#                                 width=1
#                             ),
#                             showlegend=False
#                         ))
            
#             # Add UEs
#             custom_data = []
#             for i, ue in enumerate(env.ues):
#                 bs_id = getattr(ue, 'associated_bs', -1)
#                 sinr = getattr(ue, 'sinr', -30)
#                 if sinr == -np.inf:
#                     sinr = -30
#                 custom_data.append([i, bs_id, f"{sinr:.2f}"])
            
#             fig.add_trace(go.Scatter(
#                 x=ue_coords[:, 0],
#                 y=ue_coords[:, 1],
#                 mode='markers',
#                 name='UE',
#                 marker=dict(
#                     size=10,
#                     color=[
#                         'red' if not hasattr(ue, 'associated_bs') or ue.associated_bs is None else 
#                         f'rgba(0, {min(255, int(128 + 128*(getattr(ue, "sinr", -30)+30)/60))}, 0, 0.8)' 
#                         for ue in env.ues
#                     ]
#                 ),
#                 hovertemplate="UE %{customdata[0]}<br>BS %{customdata[1]}<br>SINR %{customdata[2]}dB<extra></extra>",
#                 customdata=custom_data
#             ))
            
#             clear_and_plot(ph_marl_topo, fig, "marl_topo_count")
    
#     # Initialize HybridTraining class with MARL only
#     status_text = st.empty()
#     status_text.text("Initializing MARL training...")
    
#     hybrid = HybridTraining({
#         "env_config": {
#             "num_bs": num_bs,
#             "num_ue": num_ue,
#             "log_kpis": True
#         },
#         "marl_algorithm": "PPO",
#         "marl_steps_per_phase": marl_steps,
#         "logging": {"enabled": True},
#         "visualize_callback": visualize_marl,
#         "ray_resources": {"num_cpus": 2}
#     })
    
#     # Execute MARL without initial metaheuristic
#     status_text.text("Running MARL training...")
#     marl_result = hybrid._execute_marl_phase()
    
#     # Extract metrics from the best trial
#     best_trial = marl_result.get_best_trial("env_runners/episode_return_mean", mode="max")
#     if best_trial:
#         final_metrics = best_trial.last_result
        
#         # Display final metrics
#         st.markdown("### Final MARL Metrics")
        
#         metric_cols = st.columns(3)
#         metric_cols[0].metric("Reward", f"{final_metrics.get('env_runners/episode_return_mean', 0):.3f}")
#         metric_cols[1].metric("Average SINR", f"{final_metrics.get('custom_metrics', {}).get('average_sinr', 0):.3f} dB")
#         metric_cols[2].metric("Fairness", f"{final_metrics.get('custom_metrics', {}).get('fairness', 0):.3f}")
        
#         status_text.text("MARL training complete!")
#     else:
#         status_text.text("MARL training failed to produce results.")

# # --- IMPLEMENTATION FOR HYBRID MODE ---
# elif run and mode == "Hybrid":
#     # Create placeholders for hybrid visualizations
#     ph_hybrid_kpi = st.empty()
#     ph_hybrid_topo = ph_topo  # Use the same placeholder for topology
    
#     # Setup tracker and environment
#     tracker = KPITracker()
#     env = topo_env
    
#     # Visualization callback for hybrid approach
#     def visualize_hybrid(metrics, solution):
#         # Visualize KPI history
#         hist = tracker.history
#         if not hist.empty:
#             # Filter to only show MARL or the metaheuristic based on phase
#             meta_hist = hist[hist['phase'] == 'metaheuristic'] if 'phase' in hist.columns else pd.DataFrame()
#             marl_hist = hist[hist['phase'] == 'marl'] if 'phase' in hist.columns else pd.DataFrame()
            
#             fig_kpi = make_subplots(rows=3, cols=1, shared_xaxes=True, 
#                                   subplot_titles=["Performance", "SINR", "Fairness"])
#             fig_kpi.update_layout(height=600, margin=dict(t=50, b=40, l=40, r=40))
            
#             for i in (1, 2, 3): 
#                 fig_kpi.update_yaxes(showgrid=True, gridwidth=1, row=i, col=1)
#                 fig_kpi.update_xaxes(showgrid=True, gridwidth=1, row=i, col=1)
                
#             # Plot metaheuristic metrics if available
#             if not meta_hist.empty:
#                 if 'fitness' in meta_hist.columns:
#                     fig_kpi.add_trace(go.Scatter(
#                         x=meta_hist.index, 
#                         y=meta_hist['fitness'],
#                         name='Meta Fitness', 
#                         line=dict(width=2, color='blue')
#                     ), row=1, col=1)
                
#                 if 'average_sinr' in meta_hist.columns:
#                     fig_kpi.add_trace(go.Scatter(
#                         x=meta_hist.index, 
#                         y=meta_hist['average_sinr'],
#                         name='Meta SINR', 
#                         line=dict(width=2, color='blue')
#                     ), row=2, col=1)
                    
#                 if 'fairness' in meta_hist.columns:
#                     fig_kpi.add_trace(go.Scatter(
#                         x=meta_hist.index, 
#                         y=meta_hist['fairness'],
#                         name='Meta Fairness', 
#                         line=dict(width=2, color='blue')
#                     ), row=3, col=1)
            
#             # Plot MARL metrics if available  
#             if not marl_hist.empty:
#                 if 'reward_mean' in marl_hist.columns:
#                     # Offset MARL indices to continue after metaheuristic
#                     offset = len(meta_hist) if not meta_hist.empty else 0
#                     marl_x = [offset + i for i in range(len(marl_hist))]
                    
#                     fig_kpi.add_trace(go.Scatter(
#                         x=marl_x, 
#                         y=marl_hist['reward_mean'],
#                         name='MARL Reward', 
#                         line=dict(width=2, color='green')
#                     ), row=1, col=1)
                
#                 if 'average_sinr' in marl_hist.columns:
#                     offset = len(meta_hist) if not meta_hist.empty else 0
#                     marl_x = [offset + i for i in range(len(marl_hist))]
                    
#                     fig_kpi.add_trace(go.Scatter(
#                         x=marl_x, 
#                         y=marl_hist['average_sinr'],
#                         name='MARL SINR', 
#                         line=dict(width=2, color='green')
#                     ), row=2, col=1)
                    
#                 if 'fairness' in marl_hist.columns:
#                     offset = len(meta_hist) if not meta_hist.empty else 0
#                     marl_x = [offset + i for i in range(len(marl_hist))]
                    
#                     fig_kpi.add_trace(go.Scatter(
#                         x=marl_x, 
#                         y=marl_hist['fairness'],
#                         name='MARL Fairness', 
#                         line=dict(width=2, color='green')
#                     ), row=3, col=1)
                
#             # Display the KPI plot
#             clear_and_plot(ph_hybrid_kpi, fig_kpi, "hybrid_kpi_count")
        
#         # Visualize network topology with connections
#         if solution is not None:
#             bs_coords = np.array([b.position for b in env.base_stations])
#             ue_coords = np.array([u.position for u in env.ues])
            
#             fig = go.Figure()
            
#             # Add base stations (as icons)
#             fig.add_trace(go.Scatter(
#                 x=bs_coords[:, 0], 
#                 y=bs_coords[:, 1], 
#                 mode='markers', 
#                 name='BS', 
#                 marker=dict(size=0, color='rgba(0,0,0,0)'), 
#                 showlegend=True
#             ))
            
#             # Add BS icons
#             for i, (x, y) in enumerate(bs_coords):
#                 fig.add_layout_image(
#                     source=f"data:image/png;base64,{bs_b64}", 
#                     xref="x", 
#                     yref="y", 
#                     x=x, 
#                     y=y, 
#                     sizex=5, 
#                     sizey=5, 
#                     xanchor="center", 
#                     yanchor="middle", 
#                     layer="above"
#                 )
                
#                 # Add BS load indicators if available
#                 bs = env.base_stations[i]
#                 if hasattr(bs, 'load') and hasattr(bs, 'capacity'):
#                     load_pct = bs.load / bs.capacity if bs.capacity > 0 else 0
#                     fig.add_trace(go.Scatter(
#                         x=[x],
#                         y=[y+3],  # Position above the BS icon
#                         mode='markers',
#                         marker=dict(
#                             size=12,
#                             color=f'rgba({int(255*load_pct)}, {int(255*(1-load_pct))}, 0, 0.7)',
#                             line=dict(width=1, color='black')
#                         ),
#                         text=[f"BS {i}: {load_pct*100:.1f}%"],
#                         hoverinfo="text",
#                         showlegend=False
#                     ))
            
#             # Add connection lines between UEs and BSs based on current solution
#             if isinstance(solution, list):
#                 # If solution is a list of BS assignments (metaheuristic style)
#                 for i, bs_idx in enumerate(solution):
#                     if i < len(ue_coords) and 0 <= bs_idx < len(bs_coords):
#                         ux, uy = ue_coords[i]
#                         bx, by = bs_coords[bs_idx]
#                         fig.add_trace(go.Scatter(
#                             x=[bx, ux],
#                             y=[by, uy],
#                             mode='lines',
#                             line=dict(color='lightgray', width=1),
#                             showlegend=False
#                         ))
#             else:
#                 # If solution is based on UE objects with associated_bs attribute (MARL style)
#                 for i, ue in enumerate(env.ues):
#                     if hasattr(ue, 'associated_bs') and ue.associated_bs is not None:
#                         bs_idx = ue.associated_bs
#                         if i < len(ue_coords) and 0 <= bs_idx < len(bs_coords):
#                             ux, uy = ue_coords[i]
#                             bx, by = bs_coords[bs_idx]
#                             sinr = getattr(ue, 'sinr', 0)
#                             fig.add_trace(go.Scatter(
#                                 x=[bx, ux],
#                                 y=[by, uy],
#                                 mode='lines',
#                                 line=dict(
#                                     color=f'rgba(100, 100, 255, {max(0.2, min(1.0, (sinr+30)/60))})',
#                                     width=1
#                                 ),
#                                 showlegend=False
#                             ))
            
#             # Add UEs with custom data for hover
#             custom_data = []
#             for i, ue in enumerate(env.ues):
#                 if isinstance(solution, list) and i < len(solution):
#                     bs_id = solution[i]
#                     sinr = -30  # Default if not available
#                     if hasattr(ue, 'sinr'):
#                         sinr = ue.sinr if ue.sinr != -np.inf else -30
#                 else:
#                     bs_id = getattr(ue, 'associated_bs', -1)
#                     sinr = getattr(ue, 'sinr', -30)
#                     if sinr == -np.inf:
#                         sinr = -30
#                 custom_data.append([i, bs_id, f"{sinr:.2f}"])
            
#             # Plot UEs with color based on SINR if available
#             fig.add_trace(go.Scatter(
#                 x=ue_coords[:, 0],
#                 y=ue_coords[:, 1],
#                 mode='markers',
#                 name='UE',
#                 marker=dict(
#                     size=10,
#                     color=[
#                         'red' if len(custom_data) <= i or custom_data[i][1] == -1 else 
#                         f'rgba(0, {min(255, int(128 + 128*(float(custom_data[i][2])+30)/60))}, 0, 0.8)' 
#                         for i in range(len(ue_coords))
#                     ]
#                 ),
#                 hovertemplate="UE %{customdata[0]}<br>BS %{customdata[1]}<br>SINR %{customdata[2]}dB<extra></extra>",
#                 customdata=custom_data
#             ))
            
#             clear_and_plot(ph_hybrid_topo, fig, "hybrid_topo_count")
    
#     # Initialize and run hybrid training
#     status_text = st.empty()
#     status_text.text(f"Initializing hybrid training with {metaheuristic_algorithm} and MARL...")
    
#     hybrid = HybridTraining({
#         "env_config": {
#             "num_bs": num_bs,
#             "num_ue": num_ue,
#             "log_kpis": True
#         },
#         "marl_algorithm": "PPO",
#         "marl_steps_per_phase": marl_steps,
#         "logging": {"enabled": True},
#         "visualize_callback": visualize_hybrid,
#         "ray_resources": {"num_cpus": 2}
#     })
    
#     # Run the hybrid training process
#     status_text.text(f"Running hybrid training: {metaheuristic_algorithm} + MARL...")
#     result = hybrid.run_hybrid_training(initial_metaheuristic=metaheuristic_algorithm)
    
#     # Extract final metrics
#     final_metrics = result.get("final_metrics", {})
    
#     # Display final metrics
#     st.markdown("### Final Hybrid Training Metrics")
    
#     metric_cols = st.columns(3)
#     if "reward_mean" in final_metrics:
#         metric_cols[0].metric("Reward", f"{final_metrics.get('reward_mean', 0):.3f}")
#     else:
#         metric_cols[0].metric("Fitness", f"{final_metrics.get('fitness', 0):.3f}")
        
#     metric_cols[1].metric("Average SINR", f"{final_metrics.get('average_sinr', 0):.3f} dB")
#     metric_cols[2].metric("Fairness", f"{final_metrics.get('fairness', 0):.3f}")
    
#     # Add phase comparison if available
#     if result.get("metaheuristic_result") and result.get("marl_result"):
#         st.markdown("### Performance Comparison")
        
#         meta_metrics = result["metaheuristic_result"].get("metrics", {})
#         marl_metrics = result["marl_result"].get_best_trial("env_runners/episode_return_mean", mode="max").last_result if result["marl_result"] else {}
        
#         compare_df = pd.DataFrame({
#             "Metric": ["Performance", "SINR (dB)", "Fairness"],
#             "Metaheuristic": [
#                 f"{meta_metrics.get('fitness', 0):.3f}",
#                 f"{meta_metrics.get('average_sinr', 0):.3f}",
#                 f"{meta_metrics.get('fairness', 0):.3f}"
#             ],
#             "MARL": [
#                 f"{marl_metrics.get('env_runners/episode_return_mean', 0):.3f}",
#                 f"{marl_metrics.get('custom_metrics', {}).get('average_sinr', 0):.3f}",
#                 f"{marl_metrics.get('custom_metrics', {}).get('fairness', 0):.3f}"
#             ]
#         })
        
#         st.table(compare_df)
    
#     status_text.text("Hybrid training complete!")




# ____________________________________________________________________________________________________________________________________________




# # Add these lines before importing torch
# # Add this before any other imports
# # import os
# # os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
# import asyncio
# # Ensure there's an active event loop for Streamlit
# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())

# import streamlit as st
# import sys, os, threading, time, base64
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import copy

# # --- Load Base Station icon (before any plotting) ---
# icon_path = os.path.join(os.path.dirname(__file__), "assets", "bs_icon.png")
# with open(icon_path, "rb") as f:
#     bs_b64 = base64.b64encode(f.read()).decode()
    
# # # --- Algorithm info folder setup ---
# # algo_info_dir = os.path.join(os.path.dirname(__file__), "algo_info")  # markdown files here
# # algo_image_dir = os.path.join(os.path.dirname(__file__), "assets", "algo_images")  # images per algorithm

# import json

# info_path = os.path.join(os.path.dirname(__file__), "assets", "algo_info.json")
# with open(info_path) as f:
#     algo_info = json.load(f)


# # --- Page config ---
# st.set_page_config(page_title="6G Metaheuristic Dashboard", layout="wide")

# # --- Session-state initialization ---
# st.session_state.setdefault("cmp", {})
# for cnt in ("kpi_count","live_count","final_count","topo_single_count"):  
#     st.session_state.setdefault(cnt, 0)
# if "figures" not in st.session_state:
#     st.session_state.figures = {
#         "topo": None,
#         "kpi": None,
#         "live": None
#     }
# # Project path setup
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# sys.path.insert(0, project_root)
# from core.envs.custom_channel_env import NetworkEnvironment
# from core.hybrid_trainer.metaheuristic_opt import run_metaheuristic
# from core.hybrid_trainer.kpi_logger import KPITracker

# st.title("6G Metaheuristic Dashboard")

# # Sidebar controls
# with st.sidebar:
#     mode   = st.radio("Mode", ["Single","Comparison"] )
#     num_bs = st.slider("Base Stations", 5,50,10)
#     num_ue = st.slider("Users", 20,200,50)
#     if mode=="Single":
#         algorithm = st.selectbox("Algorithm", ["pfo","co","coa","do","fla","gto","hba","hoa","avoa","poa","rime","roa","rsa","sto"])
#     else:
#         algorithm = st.multiselect("Compare Algos", ["avoa","co","coa","do","fla","gto","hba","hoa","pfo","poa","rime","roa","rsa","sto"], default=["pfo","co"] )
#         kpi_to_compare = st.selectbox("KPI to Compare", ["fitness","average_sinr","fairness"], index=0)
#     run = st.button("Start")

# # Helper: clear & plot
# def clear_and_plot(ph, fig, counter_name, force_redraw=False):
#     # Only clears and redraws the given placeholder (used for KPI and live charts)
#     if not force_redraw and counter_name in st.session_state.figures and st.session_state.figures[counter_name] is not None:
#         st.session_state[counter_name] += 1
#         st.session_state.figures[counter_name] = copy.deepcopy(fig)
#         ph.plotly_chart(fig, use_container_width=True, key=f"{counter_name}_{st.session_state[counter_name]}")
#     else:
#         ph.empty()
#         st.session_state[counter_name] += 1
#         ph.plotly_chart(fig, use_container_width=True, key=f"{counter_name}_{st.session_state[counter_name]}")
#         st.session_state.figures[counter_name] = copy.deepcopy(fig)

# # Top row: topology + info
# col_topo, col_info = st.columns([3,1])
# with col_topo:
#     st.markdown("---")
#     ph_topo = st.expander("Network Topology", expanded=True).empty()
#     topo_env = NetworkEnvironment({"num_bs":num_bs, "num_ue":num_ue}, log_kpis=False)
#     bs_coords = np.array([b.position for b in topo_env.base_stations])
#     ue_coords = np.array([u.position for u in topo_env.ues])
#     fig_topo = go.Figure()
#     fig_topo.add_trace(go.Scatter(x=bs_coords[:,0], y=bs_coords[:,1], mode='markers', name='BS', marker=dict(size=0, color='rgba(0,0,0,0)'), showlegend=True))
#     for x,y in bs_coords:
#         fig_topo.add_layout_image(source=f"data:image/png;base64,{bs_b64}", xref="x", yref="y", x=x, y=y,
#                                   sizex=5, sizey=5, xanchor="center", yanchor="middle", layer="above")
#     fig_topo.add_trace(go.Scatter(x=ue_coords[:,0], y=ue_coords[:,1], mode='markers', name='UE', marker=dict(size=10),
#                                   hovertemplate="UE %{customdata[0]}<br>Assigned BS %{customdata[1]}<extra></extra>",
#                                   customdata=np.stack([np.arange(len(ue_coords)), [-1]*len(ue_coords)], axis=1)))
#     ph_topo.plotly_chart(fig_topo, use_container_width=True)
#     st.markdown("---")
# with col_info:
#     # with st.expander("Algorithm Info", expanded=False):
#     #     if mode=="Single":
#     #         st.write(f"**Algorithm:** {algorithm}")
#     #         st.info(f"Details about {algorithm} go here.")
#     #     else:
#     #         st.write("**Comparing:**")
#     #         for a in algorithm:
#     #             st.write(f"- **{a}**: description...")
#     #         st.info("Comparison based on selected KPI.")
    # with st.expander("Algorithm Info"):
    #     if mode == "Single":
    #         info = algo_info.get(algorithm, {})
    #         st.markdown(f"## {info.get('name', algorithm)}")
    #         st.image(info.get("image"), use_container_width=True)
    #         st.write(info.get("long", "No description available."))
    #     else:
    #         for alg in algorithm:
    #             info = algo_info.get(alg, {})
    #             st.markdown(f"**{info.get('name', alg)}**: {info.get('short','')}")
    
# # SINGLE MODE
# if run and mode=="Single":
#     ph_kpi = st.empty()
#     tracker = KPITracker()
#     env = topo_env
#     def visualize(metrics, solution):
#         hist = tracker.history
#         if not hist.empty:
#             fig_kpi = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["Fitness","SINR","Fairness"])
#             fig_kpi.update_layout(height=600, margin=dict(t=50,b=40,l=40,r=40))
#             for i in (1,2,3): fig_kpi.update_yaxes(showgrid=True, gridwidth=1, row=i, col=1); fig_kpi.update_xaxes(showgrid=True, gridwidth=1, row=i, col=1)
#             fig_kpi.add_trace(go.Scatter(x=hist.index, y=hist['fitness'], name='Fitness', line=dict(width=2)), row=1, col=1)
#             fig_kpi.add_trace(go.Scatter(x=hist.index, y=hist['average_sinr'], name='SINR', line=dict(width=2)), row=2, col=1)
#             fig_kpi.add_trace(go.Scatter(x=hist.index, y=hist['fairness'], name='Fairness', line=dict(width=2)), row=3, col=1)
#             clear_and_plot(ph_kpi, fig_kpi, "kpi_count")
#         bs_coords = np.array([b.position for b in env.base_stations])
#         ue_coords = np.array([u.position for u in env.ues])
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=bs_coords[:,0], y=bs_coords[:,1], mode='markers', name='BS', marker=dict(size=0, color='rgba(0,0,0,0)'), showlegend=True))
#         for x,y in bs_coords: fig.add_layout_image(source=f"data:image/png;base64,{bs_b64}", xref="x", yref="y", x=x, y=y, sizex=5, sizey=5, xanchor="center", yanchor="middle", layer="above")
#         for i,(ux,uy) in enumerate(ue_coords): bx,by=bs_coords[solution[i]]; fig.add_trace(go.Scatter(x=[bx,ux],y=[by,uy],mode='lines',line=dict(color='lightgray',width=1),showlegend=False))
#         custom=np.stack([np.arange(len(ue_coords)),solution],axis=1)
#         fig.add_trace(go.Scatter(x=ue_coords[:,0],y=ue_coords[:,1],mode='markers',name='UE',marker=dict(size=10),hovertemplate="UE %{customdata[0]}<br>BS %{customdata[1]}<extra></extra>",customdata=custom))
#         clear_and_plot(ph_topo, fig, "topo_single_count")
#     res = run_metaheuristic(env=env, algorithm=algorithm, epoch=0, kpi_logger=tracker, visualize_callback=visualize)
#     metrics = res["metrics"]
#     st.success("Single optimization complete!")
#     st.markdown("### Final KPIs Summary")

#     kpi_labels = {
#         "fitness": ("Fitness Score", ""),
#         "average_sinr": ("Avg SINR", "dB"),
#         "fairness": ("Fairness Index", "")
#     }
#     cols = st.columns(3)
#     for i, (kpi, (label, unit)) in enumerate(kpi_labels.items()):
#         val = metrics.get(kpi, 0)
#         cols[i].metric(label, f"{val:.3f} {unit}")

# # COMPARISON MODE
# if run and mode=="Comparison":
#     ph_live_title=st.empty(); ph_live_chart=st.empty(); ph_live_title.subheader(f"Comparison: Live {kpi_to_compare.replace('_',' ').title()}")
#     results={}; st.session_state.cmp.clear()
#     for alg in algorithm:
#         tr=KPITracker(); e=NetworkEnvironment({"num_bs":num_bs,"num_ue":num_ue},log_kpis=False); st.session_state.cmp[alg]={"tracker":tr}
#         def worker(a=alg,tr=tr,e=e): results[a]=run_metaheuristic(env=e,algorithm=a,epoch=0,kpi_logger=tr,visualize_callback=None)
#         t=threading.Thread(target=worker,daemon=True); t.start(); st.session_state.cmp[alg]["thread"]=t
#     while any(d["thread"].is_alive() for d in st.session_state.cmp.values()):
#         fig_live=make_subplots(rows=1,cols=1)
#         for alg,d in st.session_state.cmp.items():
#             h=d["tracker"].history
#             if not h.empty: fig_live.add_trace(go.Scatter(x=h.index,y=h[kpi_to_compare],name=alg))
#         clear_and_plot(ph_live_chart,fig_live,"live_count"); time.sleep(1)
#     for a,r in results.items(): st.session_state.cmp[a]["result"]=r
#     c1,_,c3=st.columns([4,1,3]);
#     with c1: st.subheader(f"Final {kpi_to_compare.replace('_',' ').title()}"); df=pd.DataFrame([{"alg":a,kpi_to_compare:r["metrics"][kpi_to_compare]} for a,r in results.items()]).set_index("alg"); st.plotly_chart(go.Figure(data=[go.Bar(x=df.index,y=df[kpi_to_compare])]),use_container_width=True)
#     with c3: 
#         st.subheader("Final KPI Summary")
#         for a,r in results.items(): st.markdown(f"### {a.upper()}"); st.write(f"{r['metrics'][kpi_to_compare]:.3f}")
#     st.success("Comparison complete!")


# # ____________________________________________________________________________________________________________


