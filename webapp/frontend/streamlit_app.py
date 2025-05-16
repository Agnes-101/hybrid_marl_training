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
for cnt in ("kpi_count", "live_count", "final_count", "topo_count", "progress_count","viz_counter", "live_specific_cmp"):
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

SCENARIOS = {
    "Small":  {"UE": [10],      "BS": [3]},
    "Medium": {"UE": [50],      "BS": [3]},
    "Large":  {"UE": [100],     "BS": [7]},
    "All":    {"UE": [10,50,100],"BS": [3,7,15]},#  [10,20,30,40,50,60,70,80,90,100]
}
# Define list of KPIs to track
all_kpis = [
            'fitness', 
            'handover_rate',
            'average_sinr', 
            'fairness', 
            'load_variance',
            'throughput',
            'energy_efficiency',
            'connection_rate'
        ]
# --- Sidebar ---
with st.sidebar:
    mode = st.radio("Mode", ["Single Metaheuristic","Custom Comparison",   
    "Specific Comparison",   "MARL", "Hybrid", "Wilcoxon Test"])
    num_bs = st.slider("Base Stations", 5, 50, 10)
    num_ue = st.slider("Users", 20, 500, 50)
    if mode in ["Single Metaheuristic", "Hybrid"]:
        meta_algo = st.selectbox("Metaheuristic Algorithm", ["pfo", "co", "coa", "do", "fla", "gto", "hba", "hoa", "avoa","aqua", "poa", "rime", "roa", "rsa", "sto"])
        iterations = st.slider("Iterations", 5, 50, 10)
    if mode == "Custom Comparison":
        iterations = st.slider("Iterations", 5, 50, 10)
        algos = st.multiselect("Compare Algos", ["avoa", "aqua","co", "coa", "do", "fla", "gto", "hba", "hoa", "pfo", "poa", "rime", "roa", "rsa", "sto"], default=["pfo", "co"])
        kpi_cmp = st.selectbox("KPI to Compare", ["fitness", "average_sinr", "fairness"])
        
    if mode == "Specific Comparison":
        iterations = st.slider("Iterations", 2, 50, 10)
        scenario_name = st.selectbox("Scenario", list(SCENARIOS.keys()))
        ue_list = SCENARIOS[scenario_name]["UE"]
        bs_list = SCENARIOS[scenario_name]["BS"]
        algos = st.multiselect("Compare Algos", ["avoa", "aqua","co", "coa", "do", "fla", "gto", "hba", "hoa", "pfo", "poa", "rime", "roa", "rsa", "sto"], default=["pfo", "co"])
        selected_kpis = st.selectbox("KPI Selection",['fitness', 'average_sinr', 'throughput', "all_kpis"], key="spec_comp_metric")        
    
    if mode in ["MARL", "Hybrid"]:
        marl_steps = st.slider("MARL Steps/Epoch", 1, 50, 10)
        # Add visualization frequency control
        viz_freq = st.slider("Visualization Frequency", 1, 20, 5, 
                             help="Update topology visualization every N steps (higher = faster but less visual feedback)")
    if mode == "Wilcoxon Test":
        iterations = st.slider("Iterations", 2, 50, 10)
        # algos = st.multiselect("Compare Algos", ["avoa", "aqua","co", "coa", "do", "fla", "gto", "hba", "hoa", "pfo", "poa", "rime", "roa", "rsa", "sto"], default=["pfo", "co"])
        # kpi_cmp = st.selectbox("KPI to Compare", ["fitness", "average_sinr", "fairness"])
    
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
        elif mode in [ "Custom Comparison","Specific Comparison"]:
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
            
        result = run_metaheuristic(env, meta_algo, epoch=0, kpi_logger=tracker, visualize_callback=viz, iterations=iterations)
        
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
    elif mode == "Custom Comparison":
        ph_live_title = st.empty()
        ph_live_chart = st.empty()
        ph_live_title.subheader(f"Custom Comparison: Live {kpi_cmp.replace('_', ' ').title()}")
        
        trackers = {}
        threads = {}
        results = {}
        env_dict = {}
        
        for a in algos:
            tr = KPITracker()
            trackers[a] = tr
            e = NetworkEnvironment({"num_bs": num_bs, "num_ue": num_ue}, log_kpis=False)
            env_dict[a] = e
            i=iterations
            def w(a=a, e=e, tr=tr): 
                results[a] = run_metaheuristic(e, a, 0, tr, None,i)
                
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
    elif mode == "Specific Comparison":
        st.header("Multi-KPI Algorithm Comparison")
        
        # 1) Pick your scenario via the dict
        scenario_name = st.selectbox("Scenario", list(SCENARIOS.keys()), key="spec_comp_scenario")
        ue_list = SCENARIOS[scenario_name]["UE"]
        bs_list = SCENARIOS[scenario_name]["BS"]      
    
        
        # Option to fix BS and vary UE only
        with st.expander("Configuration Options", expanded=True):
            vary_ue_only = st.checkbox("Vary UE only (fix BS)", value=True)
            
            if vary_ue_only:
                # If we're varying UE only, select a specific BS value
                if len(bs_list) > 1:
                    fixed_bs = st.selectbox("Fixed BS Value", bs_list, index=0)
                    bs_list = [fixed_bs]  # Replace the list with just the selected value
                else:
                    fixed_bs = bs_list[0]
                    st.info(f"Using fixed BS value: {fixed_bs}")
        
        if len(ue_list) == 1 and len(bs_list) == 1:
            # Single configuration case (live comparison)
            ue, bs = ue_list[0], bs_list[0]
            
            st.subheader(f"Live Multi-KPI Comparison @ UE={ue}, BS={bs}")
            trackers, threads, results, envs = {}, {}, {}, {}
            
            if st.button("Run Specific Comparison:"):
                for alg in algos:
                    tr = KPITracker()
                    trackers[alg] = tr
                    
                    env = NetworkEnvironment({"num_ue": ue, "num_bs": bs}, log_kpis=True)
                    envs[alg] = env
                    
                    def worker(a=alg, e=env, t=tr):
                        results[a] = run_metaheuristic(
                            e,
                            a,
                            iterations,
                            t                    
                        )
                    
                    th = threading.Thread(target=worker, daemon=True)
                    threads[alg] = th
                    th.start()
            
            # Define Plotly symbols for markers
            PLOTLY_SYMBOLS = [
                "circle", "square", "diamond", "cross", "x",
                "triangle-up", "triangle-down", "triangle-left", "triangle-right",
                "pentagon", "hexagon", "star", "hourglass", "bowtie"
            ]

            # Create mapping from algorithm name to symbol
            marker_map = {
                alg: PLOTLY_SYMBOLS[i % len(PLOTLY_SYMBOLS)]
                for i, alg in enumerate(trackers.keys())
            }
            
            # Create tabs for each KPI
            kpi_tabs = st.tabs(selected_kpis)
            placeholders = {kpi: tab.empty() for kpi, tab in zip(selected_kpis, kpi_tabs)}
            
            # Live-updating plots
            while any(t.is_alive() for t in threads.values()):
                for kpi in selected_kpis:
                    fig = make_subplots(rows=1, cols=1, subplot_titles=[f"{kpi.replace('_', ' ').title()} Progress"])
                    for alg, tr in trackers.items():
                        h = tr.history
                        if not h.empty and kpi in h:
                            fig.add_trace(go.Scatter(
                                x=h.index,
                                y=h[kpi],
                                name=alg.upper(),
                                mode="lines+markers",
                                marker=dict(symbol=marker_map[alg], size=8),
                                line=dict(dash="solid")
                            ))
                    fig.update_layout(
                        xaxis_title="Iteration",
                        yaxis_title=kpi.replace('_', ' ').title(),
                        legend_title="Algorithm",
                        height=500
                    )
                    placeholders[kpi].plotly_chart(fig, use_container_width=True)
                time.sleep(1)
                
            # Wait for any stragglers
            for t in threads.values():
                t.join()
                
            # Once all done, pull final values for each KPI
            for kpi in selected_kpis:
                final_df = pd.DataFrame([
                    {"Algorithm": a.upper(), kpi: tr.history[kpi].iloc[-1] if kpi in tr.history else None}
                    for a, tr in trackers.items()
                ]).set_index("Algorithm")
                
                # Display bar chart for this KPI
                st.subheader(f"Final {kpi.replace('_', ' ').title()} Comparison")
                st.bar_chart(final_df)
            
            st.success("Single Configuration Analysis Complete")
            
            # Create final summary table with all KPIs
            st.subheader("Final Multi-KPI Summary")
            summary_data = []
            for alg, tr in trackers.items():
                row = {"Algorithm": alg.upper()}
                for kpi in selected_kpis:
                    if kpi in tr.history:
                        row[kpi] = tr.history[kpi].iloc[-1]
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df)
            
            # Download button for multi-KPI results
            csv = summary_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"Download Multi-KPI Results",
                data=csv,
                file_name=f"multi_kpi_{scenario_name.lower()}.csv",
                mime="text/csv",
                key=f"download_multi_{scenario_name}"
            )
            
        else:
            # Multiple configurations case (batch processing)
            import itertools, pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            
            st.subheader(f"Multi-KPI UE Scaling Analysis with Fixed BS={bs_list[0]}")
            
            # Add controls for number of seeds
            n_seeds = st.slider("Number of seeds per configuration", 1, 10, 3)
            bs_list = SCENARIOS["All"]["BS"]
            selected_bs = st.selectbox("BS Configuration", bs_list, index=0)
            if selected_kpis == "all_kpis":
                selected_kpis = all_kpis
            # Calculate total runs for progress tracking
            # total_runs = len(ue_list) * len(bs_list) * len(algos) * n_seeds
            total_runs = len(ue_list) * len(algos) * n_seeds
            
            if st.button("Run Specific Comparison:"):
                st.write(f"Running {total_runs} total simulations ({len(ue_list)} UE configs × {len(algos)} algorithms × {n_seeds} seeds)")
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Store all results
                records = []
                completed_runs = 0
                
                # Run all combinations
                # for ue, bs, alg, seed_num in itertools.product(ue_list, bs_list, algos, range(1, n_seeds + 1)):
                for ue, alg, seed_num in itertools.product(ue_list, algos, range(1, n_seeds + 1)):
                    bs=selected_bs
                    # Update status
                    status_text.text(f"Running {alg.upper()} with UE={ue}, BS={bs}, seed #{seed_num}/{n_seeds}")
                    
                    # Create tracker and environment for this run
                    tr = KPITracker()
                    env = NetworkEnvironment({"num_ue": ue, "num_bs": bs})
                    
                    # Set random seed for reproducibility
                    np.random.seed(seed_num)
                    
                    # Run simulation
                    out = run_metaheuristic(
                        env=env,
                        algorithm=alg,
                        epoch=iterations,
                        kpi_logger=tr,
                        visualize_callback=None,
                        iterations=iterations
                    )
                    
                    # Get metrics using dictionary access
                    m = out["metrics"]
                    
                    # Create record with all KPIs
                    record = {
                        "UE": ue,
                        "BS": bs,
                        "Algorithm": alg.upper(),
                        "Seed": seed_num,
                        "CPU Time": m.get("cpu_time", 0)
                    }
                    
                    # Add all available KPIs to the record - handle None values
                    for kpi in selected_kpis:
                        # Ensure we have a valid value (replace None with 0 to avoid arithmetic errors)
                        record[kpi] = m.get(kpi, 0) if m.get(kpi) is not None else 0
                        
                    records.append(record)
                    
                    # Update progress
                    completed_runs += 1
                    progress_bar.progress(completed_runs / total_runs)
            
                # Create DataFrame from all results
                df_results = pd.DataFrame(records)
                
                # Display raw data if requested
                if st.checkbox("Show raw data"):
                    st.dataframe(df_results)
            
                # Aggregate statistics by UE, BS, Algorithm for all KPIs
                kpi_columns = selected_kpis + ["CPU Time"]
                agg = (
                    df_results
                    .groupby(["UE", "BS", "Algorithm"])[kpi_columns]
                    .agg(["mean", "std"])
                )
                
                # Flatten column names
                agg.columns = ["_".join(col).strip() for col in agg.columns.values]
                agg = agg.reset_index()
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    csv_raw = df_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Raw Results",
                        data=csv_raw,
                        file_name=f"{scenario_name}_raw_results.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    csv_agg = agg.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Aggregated Results",
                        data=csv_agg,
                        file_name=f"{scenario_name}_aggregated_results.csv",
                        mime="text/csv"
                    )
                
                # Create tabs for KPI visualizations
                kpi_tabs = st.tabs(selected_kpis + ["CPU Time"])
                
                # Consistent colors for algorithms
                colors = plt.cm.tab10.colors
                color_map = {alg.upper(): colors[i % len(colors)] for i, alg in enumerate(df_results["Algorithm"].unique())}
                
                # Create visualization markers
                MARKERS = ['o','s','^','D','v','>','<','p','*','h','H','X','d','+']
                marker_map = {alg.upper(): MARKERS[i % len(MARKERS)] for i, alg in enumerate(df_results["Algorithm"].unique())}
                
                # Create a plot for each KPI in its own tab
                for i, kpi in enumerate(selected_kpis + ["CPU Time"]):
                    with kpi_tabs[i]:
                        st.subheader(f"{kpi.replace('_', ' ').title()} Analysis")
                        
                        # Show scaling behavior with UE
                        fig, ax = plt.subplots(figsize=(10, 6))
                        for alg, sub in agg.groupby("Algorithm"):
                            # Sort by UE to ensure proper line drawing
                            sub = sub.sort_values("UE")
                            ax.errorbar(
                                sub["UE"],
                                sub[f"{kpi}_mean"],
                                yerr=sub[f"{kpi}_std"],  # Add error bars
                                label=alg,
                                marker=marker_map.get(alg, "o"),
                                color=color_map.get(alg, "blue"),
                                linestyle="-",
                                markersize=8,
                                capsize=4
                            )
                        
                        ax.set_xlabel("Number of UEs")
                        ax.set_ylabel(f"{kpi.replace('_', ' ').title()}")
                        ax.legend(title="Algorithm")
                        ax.grid(True, linestyle="--", alpha=0.7)
                        
                        # Add title with specific info
                        ax.set_title(f"{kpi.replace('_', ' ').title()} Scaling with UE (Fixed BS={bs_list[0]})")
                        
                        # Display plot
                        st.pyplot(fig)
                        
                        # REPLACED BAR CHARTS WITH LINE GRAPH
                        st.subheader(f"{kpi.replace('_', ' ').title()} by Algorithm at All UE Levels")
                        
                        # Create a line graph showing performance across all UE levels
                        fig, ax = plt.subplots(figsize=(12, 7))
                        
                        # Get unique algorithms and UE values
                        unique_algs = sorted(agg["Algorithm"].unique())
                        unique_ue = sorted(agg["UE"].unique())
                        
                        # For each algorithm, plot a line across all UE values
                        for alg in unique_algs:
                            alg_data = agg[agg["Algorithm"] == alg].sort_values("UE")
                            ax.plot(
                                alg_data["UE"], 
                                alg_data[f"{kpi}_mean"],
                                label=alg,
                                marker=marker_map.get(alg, "o"),
                                color=color_map.get(alg, "blue"),
                                linewidth=2,
                                markersize=8
                            )
                            
                            # Add shaded error region
                            ax.fill_between(
                                alg_data["UE"],
                                alg_data[f"{kpi}_mean"] - alg_data[f"{kpi}_std"],
                                alg_data[f"{kpi}_mean"] + alg_data[f"{kpi}_std"],
                                alpha=0.2,
                                color=color_map.get(alg, "blue")
                            )
                        
                        # Add labels and grid
                        ax.set_xlabel("Number of UEs")
                        ax.set_ylabel(f"{kpi.replace('_', ' ').title()}")
                        ax.set_title(f"{kpi.replace('_', ' ').title()} Performance Across All UE Levels")
                        ax.legend(title="Algorithm")
                        ax.grid(True, linestyle="--", alpha=0.7)
                        
                        # Set x-ticks to only show the actual UE values
                        ax.set_xticks(unique_ue)
                        
                        # Display plot
                        st.pyplot(fig)
                
                # Add a radar chart to compare algorithms across all KPIs
                st.subheader("Multi-KPI Radar Chart Comparison")
                
                # Let user select a specific UE value for the radar chart
                radar_ue = st.selectbox("Select UE for radar comparison", options=sorted(ue_list), key="radar_ue")
                
                # Filter data for the selected UE
                radar_data = agg[agg["UE"] == radar_ue]
                
                # Create a radar chart
                import matplotlib.pyplot as plt
                from matplotlib.path import Path
                from matplotlib.spines import Spine
                from matplotlib.transforms import Affine2D
                
                # Function to create a radar chart
                def radar_chart(fig, titles, values, algorithms):
                    # Number of variables
                    N = len(titles)
                    
                    # What will be the angle of each axis in the plot
                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    angles += angles[:1]  # Close the loop
                    
                    # Create subplot
                    ax = fig.add_subplot(111, polar=True)
                    
                    # Draw one axis per variable and add labels
                    plt.xticks(angles[:-1], titles, size=12)
                    
                    # Draw ylabels
                    ax.set_rlabel_position(0)
                    
                    # Plot data
                    for i, alg in enumerate(algorithms):
                        alg_values = values[i]
                        alg_values += alg_values[:1]  # Close the loop
                        ax.plot(angles, alg_values, linewidth=2, linestyle='solid', label=alg, 
                                color=color_map.get(alg, "blue"))
                        ax.fill(angles, alg_values, alpha=0.1, color=color_map.get(alg, "blue"))
                    
                    # Add legend
                    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                    
                    return ax
                
                # Normalize data for radar chart (0-1 scale for each KPI)
                radar_kpis = [kpi for kpi in selected_kpis if kpi in df_results.columns]
                
                if len(radar_kpis) >= 3:  # Need at least 3 metrics for a meaningful radar chart
                    # Create normalized data for radar chart
                    norm_data = {}
                    for kpi in radar_kpis:
                        # Get min and max values for this KPI
                        kpi_min = df_results[kpi].min()
                        kpi_max = df_results[kpi].max()
                        kpi_range = kpi_max - kpi_min if kpi_max > kpi_min else 1
                        
                        # Normalize values between 0-1
                        norm_data[kpi] = [(val - kpi_min) / kpi_range for val in radar_data[f"{kpi}_mean"]]
                    
                    # Create radar chart
                    fig = plt.figure(figsize=(10, 8))
                    algorithms = radar_data["Algorithm"].tolist()
                    
                    # Prepare data for radar chart
                    radar_values = []
                    for i, alg in enumerate(algorithms):
                        alg_values = [norm_data[kpi][i] for kpi in radar_kpis]
                        radar_values.append(alg_values)
                    
                    # Create the radar chart
                    ax = radar_chart(fig, radar_kpis, radar_values, algorithms)
                    ax.set_title(f"Algorithm Comparison Across All KPIs (UE={radar_ue}, BS={bs_list[0]})")
                    
                    # Display the plot
                    st.pyplot(fig)
                else:
                    st.warning("Need at least 3 KPIs for radar chart visualization")
                
                # Add a correlation heatmap between KPIs
                st.subheader("KPI Correlation Analysis")
                
                # Compute correlation matrix between KPIs
                corr_columns = selected_kpis + ["CPU Time"]
                corr_df = df_results[corr_columns].corr()
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(corr_df, cmap="coolwarm")
                
                # Add colorbar
                cbar = ax.figure.colorbar(im, ax=ax)
                
                # Set tick labels
                ax.set_xticks(np.arange(len(corr_columns)))
                ax.set_yticks(np.arange(len(corr_columns)))
                ax.set_xticklabels(corr_columns, rotation=45, ha="right")
                ax.set_yticklabels(corr_columns)
                
                # Add correlation values in the cells
                for i in range(len(corr_columns)):
                    for j in range(len(corr_columns)):
                        text = ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}",
                                    ha="center", va="center", color="black" if abs(corr_df.iloc[i, j]) < 0.7 else "white")
                
                ax.set_title("Correlation Between KPIs")
                fig.tight_layout()
                
                # Display the plot
                st.pyplot(fig)
                
                # Show success message
                st.success(f"Multi-KPI analysis completed for {len(ue_list)} UE configurations with fixed BS={bs_list[0]}")   
    # elif mode == "Specific Comparison":
    #     st.header(f"Specific Comparison: {metric.replace('_',' ').title()}")
    #     # 1) Pick your scenario via the dict
    #     scenario_name = st.selectbox("Scenario", list(SCENARIOS.keys()), key="spec_comp_scenario")
    #     ue_list = SCENARIOS[scenario_name]["UE"]
    #     bs_list = SCENARIOS[scenario_name]["BS"]      
        
    #     if len(ue_list) == 1 and len(bs_list) == 1:
    #         ue, bs = ue_list[0], bs_list[0]
            
    #         st.subheader(f"Live Comparison @ UE={ue}, BS={bs}")
    #         trackers, threads, results, envs = {}, {}, {}, {}
            
    #         for alg in algos:
    #             tr = KPITracker()
    #             trackers[alg] = tr
                
    #             env = NetworkEnvironment({"num_ue": ue, "num_bs": bs}, log_kpis=True)
    #             envs[alg] = env
                
    #             def worker(a=alg, e=env, t=tr):
    #                 results[a] = run_metaheuristic(
    #                 e,
    #                 a,
    #                 epoch=iterations,
    #                 kpi_logger=t,
    #                 iterations=iterations,
    #                 visualize_callback=None                    
    #                 )
                
    #             th = threading.Thread(target=worker, daemon=True)
    #             threads[alg] = th
    #             th.start()
    #         # 1) Define a palette of Plotly symbols
    #         PLOTLY_SYMBOLS = [
    #             "circle", "square", "diamond", "cross", "x",
    #             "triangle-up", "triangle-down", "triangle-left", "triangle-right",
    #             "pentagon", "hexagon", "star", "hourglass", "bowtie"
    #         ]

    #         # 2) Create a mapping from algorithm name → Plotly symbol
    #         marker_map = {
    #             alg: PLOTLY_SYMBOLS[i % len(PLOTLY_SYMBOLS)]
    #             for i, alg in enumerate(trackers.keys())
    #         }
    #         # # live‐updating chart
    #         # placeholder = st.empty()
    #         # while any(t.is_alive() for t in threads.values()):
    #         #     fig = make_subplots(rows=1, cols=1)
    #         #     for a, tr in trackers.items():
    #         #         h = tr.history
    #         #         if not h.empty and metric in h:
    #         #             fig.add_trace(go.Scatter(
    #         #                 x=h.index, y=h[metric], name=a.upper(), mode="lines+markers"
    #         #             ))
    #         #     clear_and_plot(placeholder, fig, "live_specific_cmp")
    #         #     time.sleep(1)
    #         # 3) Live‐updating plot
    #         placeholder = st.empty()
    #         while any(t.is_alive() for t in threads.values()):
    #             fig = make_subplots(rows=1, cols=1)
    #             for alg, tr in trackers.items():
    #                 h = tr.history
    #                 if not h.empty and metric in h:
    #                     fig.add_trace(go.Scatter(
    #                         x=h.index,
    #                         y=h[metric],
    #                         name=alg.upper(),
    #                         mode="lines+markers",
    #                         marker=dict(symbol=marker_map[alg], size=8),
    #                         line=dict(dash="solid")
    #                     ))
    #             clear_and_plot(placeholder, fig, "live_specific_cmp")
    #             time.sleep(1)
    #         # wait for any stragglers
    #         for t in threads.values():
    #             t.join()
    #         # once all done, pull final values from each tracker
    #         final_df = pd.DataFrame([
    #             {"Algorithm": a.upper(), metric: tr.history[metric].iloc[-1]}
    #             for a, tr in trackers.items()
    #         ]).set_index("Algorithm")
    #         st.bar_chart(final_df)
            
    #         st.success("Specific Comparison Complete")
    #        # Download button for this single scenario
    #         csv = final_df.reset_index().to_csv(index=False).encode("utf-8")
    #         st.download_button(
    #             f"Download {scenario_name} Scenario Results",
    #             data=csv,
    #             file_name=f"results_{scenario_name.lower()}.csv",
    #             mime="text/csv",
    #             key=f"download_{scenario_name}"
    #         )

    #     else:
    #         # Multiple configurations case (batch processing)
    #         import itertools, pandas as pd
    #         import numpy as np
    #         import matplotlib.pyplot as plt
            
    #         st.subheader(f"UE Scaling Analysis with Fixed BS={bs_list[0]}")
            
    #         # Add controls for number of seeds
    #         n_seeds = st.slider("Number of seeds per configuration", 1, 10, 3)
            
    #         # Calculate total runs for progress tracking
    #         total_runs = len(ue_list) * len(bs_list) * len(algos) * n_seeds
    #         st.write(f"Running {total_runs} total simulations ({len(ue_list)} UE configs × {len(bs_list)} BS configs × {len(algos)} algorithms × {n_seeds} seeds)")
            
    #         # Create progress bar
    #         progress_bar = st.progress(0)
    #         status_text = st.empty()
            
    #         # Store all results
    #         records = []
    #         completed_runs = 0
            
    #         # Run all combinations
    #         for ue, bs, alg, seed_num in itertools.product(ue_list, bs_list, algos, range(1, n_seeds + 1)):
    #             # Update status
    #             status_text.text(f"Running {alg.upper()} with UE={ue}, BS={bs}, seed #{seed_num}/{n_seeds}")
                
    #             # Create tracker and environment for this run
    #             tr = KPITracker()
    #             env = NetworkEnvironment({"num_ue": ue, "num_bs": bs})
                
    #             # Set random seed for reproducibility
    #             np.random.seed(seed_num)
                
    #             # Run simulation
    #             out = run_metaheuristic(
    #                 env=env,
    #                 algorithm=alg,
    #                 epoch=iterations,
    #                 kpi_logger=tr,
    #                 visualize_callback=None,
    #                 iterations=iterations
    #             )
                
    #             # Get metrics using dictionary access
    #             m = out["metrics"]
                
    #             # Record results
    #             records.append({
    #                 "UE": ue,
    #                 "BS": bs,
    #                 "Algorithm": alg.upper(),
    #                 "Seed": seed_num,
    #                 metric: m.get(metric),
    #                 "CPU Time": m.get("cpu_time"),
    #             })
                
    #             # Update progress
    #             completed_runs += 1
    #             progress_bar.progress(completed_runs / total_runs)
            
    #         # Create DataFrame from all results
    #         df_results = pd.DataFrame(records)
            
    #         # Display raw data if requested
    #         if st.checkbox("Show raw data"):
    #             st.dataframe(df_results)
            
    #         # Aggregate statistics by UE, BS, Algorithm
    #         agg = (
    #             df_results
    #             .groupby(["UE", "BS", "Algorithm"])[[metric, "CPU Time"]]
    #             .agg(["mean", "std"])
    #         )
            
    #         # Flatten column names
    #         agg.columns = ["_".join(col).strip() for col in agg.columns.values]
    #         agg = agg.reset_index()
            
    #         # Download buttons
    #         col1, col2 = st.columns(2)
    #         with col1:
    #             csv_raw = df_results.to_csv(index=False).encode("utf-8")
    #             st.download_button(
    #                 label="Download Raw Results",
    #                 data=csv_raw,
    #                 file_name=f"{scenario_name}_raw_results.csv",
    #                 mime="text/csv"
    #             )
            
    #         with col2:
    #             csv_agg = agg.to_csv(index=False).encode("utf-8")
    #             st.download_button(
    #                 label="Download Aggregated Results",
    #                 data=csv_agg,
    #                 file_name=f"{scenario_name}_aggregated_results.csv",
    #                 mime="text/csv"
    #             )
            
    #         # Create tabs for different visualizations
    #         tab1, tab2 = st.tabs(["UE Scaling Performance", "Algorithm Comparison"])
            
    #         with tab1:
    #             st.subheader(f"Algorithm Performance vs Number of UEs (Fixed BS={bs_list[0]})")
                
    #             # Consistent colors for algorithms
    #             colors = plt.cm.tab10.colors
    #             color_map = {alg.upper(): colors[i % len(colors)] for i, alg in enumerate(df_results["Algorithm"].unique())}
                
    #             # Create visualization
    #             MARKERS = ['o','s','^','D','v','>','<','p','*','h','H','X','d','+']
    #             marker_map = {alg.upper(): MARKERS[i % len(MARKERS)] for i, alg in enumerate(df_results["Algorithm"].unique())}

    #             # Draw plot
    #             fig, ax = plt.subplots(figsize=(10, 6))
    #             for alg, sub in agg.groupby("Algorithm"):
    #                 # Sort by UE to ensure proper line drawing
    #                 sub = sub.sort_values("UE")
    #                 ax.errorbar(
    #                     sub["UE"],
    #                     sub[f"{metric}_mean"],
    #                     yerr=sub[f"{metric}_std"],  # Add error bars
    #                     label=alg,
    #                     marker=marker_map.get(alg, "o"),
    #                     color=color_map.get(alg, "blue"),
    #                     linestyle="-",
    #                     markersize=8,
    #                     capsize=4
    #                 )
                
    #             ax.set_xlabel("Number of UEs")
    #             ax.set_ylabel(f"{metric.replace('_', ' ').title()}")
    #             ax.legend(title="Algorithm")
    #             ax.grid(True, linestyle="--", alpha=0.7)
                
    #             # Add title with specific info
    #             ax.set_title(f"Algorithm Scaling Performance (Fixed BS={bs_list[0]})")
                
    #             # Display plot
    #             st.pyplot(fig)
                
    #         with tab2:
    #             st.subheader("Comparison of Algorithm Performance")
                
    #             # Create a bar chart for each UE value
    #             for ue in sorted(ue_list):
    #                 # Filter data for this UE
    #                 ue_data = agg[agg["UE"] == ue]
                    
    #                 # Sort by performance (assuming higher is better)
    #                 ue_data = ue_data.sort_values(f"{metric}_mean", ascending=False)
                    
    #                 # Create bar chart
    #                 fig, ax = plt.subplots(figsize=(10, 5))
    #                 bars = ax.bar(
    #                     ue_data["Algorithm"],
    #                     ue_data[f"{metric}_mean"],
    #                     yerr=ue_data[f"{metric}_std"],
    #                     capsize=4,
    #                     color=[color_map.get(alg, "blue") for alg in ue_data["Algorithm"]]
    #                 )
                    
    #                 # Add labels
    #                 ax.set_xlabel("Algorithm")
    #                 ax.set_ylabel(f"{metric.replace('_', ' ').title()}")
    #                 ax.set_title(f"Algorithm Performance at UE={ue}, BS={bs_list[0]}")
                    
    #                 # Add value labels on top of bars
    #                 for bar in bars:
    #                     height = bar.get_height()
    #                     ax.text(
    #                         bar.get_x() + bar.get_width()/2.,
    #                         height + 0.02,
    #                         f'{height:.2f}',
    #                         ha='center', va='bottom', rotation=0
    #                     )
                    
    #                 # Display plot
    #                 st.pyplot(fig)
                    
    #             # Add CPU time comparison
    #             st.subheader("Algorithm Execution Time")
                
    #             # Create box plot of CPU times by algorithm
    #             fig, ax = plt.subplots(figsize=(10, 6))
                
    #             # Create box plot data
    #             box_data = []
    #             labels = []
    #             for alg in df_results["Algorithm"].unique():
    #                 alg_data = df_results[df_results["Algorithm"] == alg]["CPU Time"]
    #                 box_data.append(alg_data)
    #                 labels.append(alg)
                
    #             # Create the box plot
    #             ax.boxplot(box_data, labels=labels, patch_artist=True)
                
    #             # Add labels
    #             ax.set_xlabel("Algorithm")
    #             ax.set_ylabel("CPU Time (seconds)")
    #             ax.set_title("Distribution of Algorithm Execution Times")
    #             ax.grid(True, linestyle="--", alpha=0.7)
                
    #             # Display plot
    #             st.pyplot(fig)
            
    #         # Show success message
    #         st.success(f"Analysis completed for {len(ue_list)} UE configurations with fixed BS={bs_list[0]}")
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
            meta_result = run_metaheuristic(env, meta_algo, epoch=0, kpi_logger=tracker, visualize_callback=viz_meta, iterations=iterations)
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
                
    elif mode == "Wilcoxon Test":
        st.header("Statistical Algorithm Comparison")
        
        # Configure test parameters
        # iterations = st.slider("Iterations per run", 10, 100, 30)
        n_seeds = st.slider("Number of seeds (samples)", 2, 30, 10)
        
        # Choose baseline algorithm (default to PFO)
        baseline_algo = st.selectbox(
            "Baseline Algorithm", 
            ["pfo", "avoa", "aqua", "co", "coa", "do", "fla", "gto", "hba", "hoa", "poa", "rime", "roa", "rsa", "sto"],
            index=0  # Default to PFO
        )
        
        # Choose algorithms to compare against baseline
        comparison_algos = st.multiselect(
            "Algorithms to Compare Against Baseline", 
            [algo for algo in ["avoa", "aqua", "co", "coa", "do", "fla", "gto", "hba", "hoa", "pfo", "poa", "rime", "roa", "rsa", "sto"] if algo != baseline_algo],
            default=["co", "rime"]  # Default selection
        )
        
        # Add option to compare all algorithms against baseline
        if st.checkbox("Compare All Algorithms Against Baseline", value=False):
            comparison_algos = [algo for algo in ["avoa", "aqua", "co", "coa", "do", "fla", "gto", "hba", "hoa", "pfo", "poa", "rime", "roa", "rsa", "sto"] 
                            if algo != baseline_algo]
            st.info(f"Comparing {baseline_algo.upper()} against {len(comparison_algos)} algorithms")
        
        # Choose KPI to compare
        kpi_to_compare = st.selectbox(
            "KPI to Compare", 
            ["fitness", "average_sinr", "fairness", "energy_efficiency", "spectral_efficiency", "coverage", "load_balance"],
            index=0
        )
        
        # Select scenario
        scenario_name = st.selectbox("Scenario", list(SCENARIOS.keys()), key="wilcoxon_scenario")
        ue_list = SCENARIOS[scenario_name]["UE"]
        bs_list = SCENARIOS[scenario_name]["BS"]
        
        # Choose specific UE/BS config for the test
        selected_ue = st.selectbox("UE Configuration", ue_list, index=0)
        selected_bs = st.selectbox("BS Configuration", bs_list, index=0)
        
        # Run button
        if st.button("Run Wilcoxon Test"):
            # Import required packages
            from scipy import stats
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Calculate total runs
            total_runs = (len(comparison_algos) + 1) * n_seeds  # +1 for baseline
            completed_runs = 0
            
            # Dictionary to store results
            all_results = {baseline_algo: []}
            for algo in comparison_algos:
                all_results[algo] = []
            
            # Run baseline algorithm first
            status_text.text(f"Running baseline {baseline_algo.upper()} algorithm...")
            
            for seed in range(1, n_seeds + 1):
                # Update status
                status_text.text(f"Running {baseline_algo.upper()} (seed {seed}/{n_seeds})")
                
                # Set random seed for reproducibility
                np.random.seed(seed)
                
                # Create environment and tracker
                env = NetworkEnvironment({"num_ue": selected_ue, "num_bs": selected_bs})
                tracker = KPITracker()
                
                # Run algorithm
                result = run_metaheuristic(
                    env=env,
                    algorithm=baseline_algo,
                    epoch=iterations,
                    kpi_logger=tracker,
                    visualize_callback=None,
                    iterations=iterations
                )
                
                # Extract the KPI value for this run
                kpi_value = result["metrics"].get(kpi_to_compare, 0)
                if kpi_value is None:
                    kpi_value = 0
                
                # Store result
                all_results[baseline_algo].append({
                    "algorithm": baseline_algo.upper(),
                    "seed": seed,
                    "kpi_value": kpi_value,
                    "cpu_time": result["metrics"].get("cpu_time", 0)
                })
                
                # Update progress
                completed_runs += 1
                progress_bar.progress(completed_runs / total_runs)
            
            # Run comparison algorithms
            for algo in comparison_algos:
                for seed in range(1, n_seeds + 1):
                    # Update status
                    status_text.text(f"Running {algo.upper()} (seed {seed}/{n_seeds})")
                    
                    # Set random seed for reproducibility
                    np.random.seed(seed)
                    
                    # Create environment and tracker
                    env = NetworkEnvironment({"num_ue": selected_ue, "num_bs": selected_bs})
                    tracker = KPITracker()
                    
                    # Run algorithm
                    result = run_metaheuristic(
                        env=env,
                        algorithm=algo,
                        epoch=iterations,
                        kpi_logger=tracker,
                        visualize_callback=None,
                        iterations=iterations
                    )
                    
                    # Extract the KPI value for this run
                    kpi_value = result["metrics"].get(kpi_to_compare, 0)
                    if kpi_value is None:
                        kpi_value = 0
                    
                    # Store result
                    all_results[algo].append({
                        "algorithm": algo.upper(),
                        "seed": seed,
                        "kpi_value": kpi_value,
                        "cpu_time": result["metrics"].get("cpu_time", 0)
                    })
                    
                    # Update progress
                    completed_runs += 1
                    progress_bar.progress(completed_runs / total_runs)
            
            # Test complete - create DataFrame with all results
            records = []
            for algo, results in all_results.items():
                records.extend(results)
            
            results_df = pd.DataFrame(records)
            
            # Display raw results if requested
            if st.checkbox("Show raw results", value=False):
                st.dataframe(results_df)
            
            # Perform Wilcoxon signed-rank test for each algorithm against baseline
            st.subheader(f"Wilcoxon Signed-Rank Test Results (vs. {baseline_algo.upper()})")
            
            wilcoxon_results = []
            baseline_values = np.array([r["kpi_value"] for r in all_results[baseline_algo]])
            
            for algo in comparison_algos:
                algo_values = np.array([r["kpi_value"] for r in all_results[algo]])
                
                # Perform Wilcoxon test
                w_stat, p_value = stats.wilcoxon(baseline_values, algo_values)
                
                # Calculate effect size - Cliff's Delta is a good non-parametric effect size
                # (simplified calculation here)
                mean_diff = np.mean(algo_values) - np.mean(baseline_values)
                pooled_std = np.sqrt((np.std(baseline_values)**2 + np.std(algo_values)**2) / 2)
                effect_size = mean_diff / pooled_std if pooled_std != 0 else 0
                
                # Determine significance level
                if p_value < 0.01:
                    sig_level = "*** (p<0.01)"
                elif p_value < 0.05:
                    sig_level = "** (p<0.05)"
                elif p_value < 0.1:
                    sig_level = "* (p<0.1)"
                else:
                    sig_level = "ns"
                
                # Determine which algorithm is better
                baseline_mean = np.mean(baseline_values)
                algo_mean = np.mean(algo_values)
                
                # For fairness and most KPIs, higher is better, but note this might need adjustment
                # for KPIs where lower is better
                comparison = "better" if algo_mean > baseline_mean else "worse"
                if abs(algo_mean - baseline_mean) / max(baseline_mean, 1e-10) < 0.01:  # 1% threshold
                    comparison = "similar"
                    
                # Add to results
                wilcoxon_results.append({
                    "Algorithm": algo.upper(),
                    "vs. Baseline": baseline_algo.upper(),
                    "Mean Difference": algo_mean - baseline_mean,
                    "% Difference": ((algo_mean - baseline_mean) / max(baseline_mean, 1e-10)) * 100,
                    "p-value": p_value,
                    "Significance": sig_level,
                    "Effect Size": effect_size,
                    "Comparison": comparison
                })
            
            # Create and display Wilcoxon test results table
            wilcoxon_df = pd.DataFrame(wilcoxon_results)
            st.dataframe(wilcoxon_df)
            
            # Create summary visualization
            st.subheader(f"Statistical Comparison for {kpi_to_compare.replace('_', ' ').title()}")
            
            # Boxplot comparison
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Prepare data for boxplot
            box_data = []
            box_labels = []
            
            # Always put baseline first
            box_data.append([r["kpi_value"] for r in all_results[baseline_algo]])
            box_labels.append(f"{baseline_algo.upper()} (Baseline)")
            
            for algo in comparison_algos:
                box_data.append([r["kpi_value"] for r in all_results[algo]])
                box_labels.append(algo.upper())
            
            # Create boxplot
            bp = ax.boxplot(box_data, patch_artist=True, labels=box_labels)
            
            # Color the baseline differently
            for i, box in enumerate(bp['boxes']):
                if i == 0:  # Baseline
                    box.set(facecolor='lightblue')
                else:
                    # Color based on significance
                    if wilcoxon_results[i-1]["p-value"] < 0.05:
                        if wilcoxon_results[i-1]["Comparison"] == "better":
                            box.set(facecolor='lightgreen')
                        elif wilcoxon_results[i-1]["Comparison"] == "worse":
                            box.set(facecolor='lightcoral')
                        else:
                            box.set(facecolor='lightyellow')
                    else:
                        box.set(facecolor='lightyellow')
            
            # Add title and labels
            ax.set_title(f'Distribution of {kpi_to_compare.replace("_", " ").title()} Values')
            ax.set_ylabel(kpi_to_compare.replace('_', ' ').title())
            ax.set_xlabel('Algorithm')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Add a legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightblue', label='Baseline'),
                Patch(facecolor='lightgreen', label='Significantly Better (p<0.05)'),
                Patch(facecolor='lightcoral', label='Significantly Worse (p<0.05)'),
                Patch(facecolor='lightyellow', label='No Significant Difference')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            # Add grid for easier reading
            ax.grid(True, linestyle='--', alpha=0.7)
            
            fig.tight_layout()
            st.pyplot(fig)
            
            # Create bar chart showing mean values with error bars (95% CI)
            st.subheader(f"Mean {kpi_to_compare.replace('_', ' ').title()} Comparison")
            
            # Calculate means and confidence intervals
            means = []
            ci_low = []
            ci_high = []
            labels = []
            
            # Always put baseline first
            baseline_data = np.array([r["kpi_value"] for r in all_results[baseline_algo]])
            means.append(np.mean(baseline_data))
            # 95% CI using t-distribution
            sem = stats.sem(baseline_data)
            ci = sem * stats.t.ppf((1 + 0.95) / 2, len(baseline_data) - 1)
            ci_low.append(means[0] - ci)
            ci_high.append(means[0] + ci)
            labels.append(f"{baseline_algo.upper()} (Baseline)")
            
            for algo in comparison_algos:
                algo_data = np.array([r["kpi_value"] for r in all_results[algo]])
                means.append(np.mean(algo_data))
                sem = stats.sem(algo_data)
                ci = sem * stats.t.ppf((1 + 0.95) / 2, len(algo_data) - 1)
                ci_low.append(means[-1] - ci)
                ci_high.append(means[-1] + ci)
                labels.append(algo.upper())
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot bars
            x = np.arange(len(labels))
            bars = ax.bar(x, means, yerr=np.array([np.array(means) - np.array(ci_low), 
                                                np.array(ci_high) - np.array(means)]),
                        capsize=5, alpha=0.7)
            
            # Color bars based on statistical significance
            bars[0].set_color('lightblue')  # Baseline
            
            for i in range(1, len(bars)):
                if wilcoxon_results[i-1]["p-value"] < 0.05:
                    if wilcoxon_results[i-1]["Comparison"] == "better":
                        bars[i].set_color('lightgreen')
                    elif wilcoxon_results[i-1]["Comparison"] == "worse":
                        bars[i].set_color('lightcoral')
                    else:
                        bars[i].set_color('lightyellow')
                else:
                    bars[i].set_color('lightyellow')
            
            # Add labels and title
            ax.set_ylabel(kpi_to_compare.replace('_', ' ').title())
            ax.set_title(f'Mean {kpi_to_compare.replace("_", " ").title()} with 95% Confidence Intervals')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
            # Add significance markers
            for i in range(1, len(bars)):
                if wilcoxon_results[i-1]["p-value"] < 0.01:
                    marker = "***"
                elif wilcoxon_results[i-1]["p-value"] < 0.05:
                    marker = "**"
                elif wilcoxon_results[i-1]["p-value"] < 0.1:
                    marker = "*"
                else:
                    marker = ""
                    
                if marker:
                    height = max(means[i], means[0]) * 1.05
                    ax.text(x[i], height, marker, ha='center', va='bottom', fontsize=12)
            
            # Add a legend
            legend_elements = [
                Patch(facecolor='lightblue', label='Baseline'),
                Patch(facecolor='lightgreen', label='Significantly Better (p<0.05)'),
                Patch(facecolor='lightcoral', label='Significantly Worse (p<0.05)'),
                Patch(facecolor='lightyellow', label='No Significant Difference')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            # Add significance legend
            ax.text(0.98, 0.02, "*** p<0.01, ** p<0.05, * p<0.1", 
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=10)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3, axis='y')
            
            fig.tight_layout()
            st.pyplot(fig)
            
            # Create convergence plot to compare performance over iterations
            st.subheader("Convergence Behavior Analysis")
            
            # First, rerun algorithms to track convergence
            convergence_data = {algo: [] for algo in [baseline_algo] + comparison_algos}
            
            # Select a single seed for convergence analysis
            convergence_seed = n_seeds // 2  # Middle seed
            
            # Create placeholder for convergence plot
            convergence_plot = st.empty()
            
            # Run baseline and comparison algos and track convergence
            for algo in [baseline_algo] + comparison_algos:
                # Set random seed
                np.random.seed(convergence_seed)
                
                # Create environment
                env = NetworkEnvironment({"num_ue": selected_ue, "num_bs": selected_bs})
                
                # Create tracker that will record history
                tracker = KPITracker()
                
                # Run algorithm with the tracker
                result = run_metaheuristic(
                    env=env,
                    algorithm=algo,
                    epoch=iterations,
                    kpi_logger=tracker,
                    visualize_callback=None,
                    iterations=iterations
                )
                
                # Store convergence data
                if kpi_to_compare in tracker.history:
                    convergence_data[algo] = tracker.history[kpi_to_compare].tolist()
                else:
                    # If KPI not in history, create a flat line with the final value
                    convergence_data[algo] = [result["metrics"].get(kpi_to_compare, 0)] * iterations
            
            # Create convergence plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot line for baseline
            ax.plot(range(1, len(convergence_data[baseline_algo])+1), 
                    convergence_data[baseline_algo], 
                    label=f"{baseline_algo.upper()} (Baseline)",
                    linewidth=3, color='blue')
            
            # Plot lines for comparison algorithms
            colors = plt.cm.tab10.colors
            for i, algo in enumerate(comparison_algos):
                ax.plot(range(1, len(convergence_data[algo])+1), 
                        convergence_data[algo],
                        label=algo.upper(),
                        linewidth=1.5, color=colors[(i+1) % len(colors)])
            
            # Add labels and title
            ax.set_xlabel('Iteration')
            ax.set_ylabel(kpi_to_compare.replace('_', ' ').title())
            ax.set_title(f'Convergence Behavior for {kpi_to_compare.replace("_", " ").title()}')
            
            # Add legend
            ax.legend()
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Display convergence plot
            convergence_plot.pyplot(fig)
            
            # Create summary table
            st.subheader("Statistical Summary")
            
            # Calculate summary statistics
            summary_data = []
            
            # Baseline first
            baseline_data = np.array([r["kpi_value"] for r in all_results[baseline_algo]])
            summary_data.append({
                "Algorithm": f"{baseline_algo.upper()} (Baseline)",
                "Mean": np.mean(baseline_data),
                "Median": np.median(baseline_data),
                "Std Dev": np.std(baseline_data),
                "Min": np.min(baseline_data),
                "Max": np.max(baseline_data),
                "95% CI Lower": np.mean(baseline_data) - stats.sem(baseline_data) * stats.t.ppf((1 + 0.95) / 2, len(baseline_data) - 1),
                "95% CI Upper": np.mean(baseline_data) + stats.sem(baseline_data) * stats.t.ppf((1 + 0.95) / 2, len(baseline_data) - 1)
            })
            
            # Comparison algorithms
            for algo in comparison_algos:
                algo_data = np.array([r["kpi_value"] for r in all_results[algo]])
                summary_data.append({
                    "Algorithm": algo.upper(),
                    "Mean": np.mean(algo_data),
                    "Median": np.median(algo_data),
                    "Std Dev": np.std(algo_data),
                    "Min": np.min(algo_data),
                    "Max": np.max(algo_data),
                    "95% CI Lower": np.mean(algo_data) - stats.sem(algo_data) * stats.t.ppf((1 + 0.95) / 2, len(algo_data) - 1),
                    "95% CI Upper": np.mean(algo_data) + stats.sem(algo_data) * stats.t.ppf((1 + 0.95) / 2, len(algo_data) - 1)
                })
            
            # Display summary table
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df)
            
            # Download buttons for results
            st.subheader("Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_raw = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Raw Results",
                    data=csv_raw,
                    file_name=f"wilcoxon_raw_results_{kpi_to_compare}.csv",
                    mime="text/csv"
                )
                # Add view toggle button
                if st.button("📊 View Raw", key="view_raw"):
                    st.session_state['show_raw'] = not st.session_state.get('show_raw', False)
            with col2:
                csv_wilcoxon = wilcoxon_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Wilcoxon Results",
                    data=csv_wilcoxon,
                    file_name=f"wilcoxon_test_results_{kpi_to_compare}.csv",
                    mime="text/csv"
                )
                # Add view toggle button
                if st.button("📊 View Wilcoxon", key="view_wilcoxon"):
                    st.session_state['show_wilcoxon'] = not st.session_state.get('show_wilcoxon', False)
                    
            with col3:
                csv_summary = summary_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Summary Statistics",
                    data=csv_summary,
                    file_name=f"wilcoxon_summary_{kpi_to_compare}.csv",
                    mime="text/csv"
                )
                # Add view toggle button
                if st.button("📊 View Summary", key="view_summary"):
                    st.session_state['show_summary'] = not st.session_state.get('show_summary', False)
            # Display viewed results based on toggle states
            if st.session_state.get('show_raw', False):
                st.subheader("Raw Results Preview")
                st.dataframe(results_df.head(20))
                
            if st.session_state.get('show_wilcoxon', False):
                st.subheader("Wilcoxon Test Results Preview")
                st.dataframe(wilcoxon_df)
                
            if st.session_state.get('show_summary', False):
                st.subheader("Summary Statistics Preview")
                st.dataframe(summary_df)        
            # Final success message
            st.success(f"Wilcoxon test completed comparing {baseline_algo.upper()} against {len(comparison_algos)} algorithms for {kpi_to_compare} KPI")    
