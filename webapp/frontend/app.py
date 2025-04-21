# webapp/frontend/app.py
import streamlit as st
import requests

def main():
    st.title("6G Optimization Dashboard")
    
    # Control Panel
    with st.sidebar:
        algorithm = st.selectbox("Algorithm", ["pfo", "pso", "aco"])
        num_bs = st.slider("Base Stations", 1, 50, 10)
        num_ue = st.slider("User Equipment", 10, 300, 50)
        
        if st.button("Start Optimization"):
            response = requests.post(
                "http://localhost:8000/start",
                json={"algorithm": algorithm, "num_bs": num_bs, "num_ue":num_ue}
            )
            task_id = response.json()["task_id"]
            st.session_state.task_id = task_id

    # Visualization Area
    if "task_id" in st.session_state:
        st.write(f"Running Task: {st.session_state.task_id}")
        
if __name__ == "__main__":
    main()