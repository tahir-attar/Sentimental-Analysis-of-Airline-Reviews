import streamlit as st
import pandas as pd
import json
import os
import time
from datetime import datetime
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# ------------- CONFIG -------------
CSV_FILE = "airline_reviews_with_aspect_sentiments.csv"
STATE_FILE = "state.json"
REFRESH_INTERVAL_SEC = 1  # Auto-refresh every N seconds
ASPECTS = ["Seat Comfort", "Food Quality", "Staff Service", "Cleanliness", "Entertainment"]  # Example aspects
# -----------------------------------

# --- Helper: Load or initialize state ---
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"last_index": -1, "aspect_scores": {a: [] for a in ASPECTS}, "timestamps": []}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

# --- Helper: Load reviews dataset ---
@st.cache_data
def load_reviews():
    df = pd.read_csv(CSV_FILE)
    return df

# --- Process next review in sequence ---
def process_next_review(state, df):
    next_index = state["last_index"] + 1
    if next_index < len(df):
        row = df.iloc[next_index]
        
        # Simulated aspect sentiment update
        for aspect in ASPECTS:
            # If the CSV contains scores for each aspect, use them; else randomize
            if aspect in row:
                score = row[aspect]
            else:
                import random
                score = random.uniform(0, 100)  # Placeholder if aspect score missing
            state["aspect_scores"][aspect].append(score)
        
        state["timestamps"].append(datetime.now().strftime("%H:%M:%S"))
        state["last_index"] = next_index
        save_state(state)

# --- Plot radial gauges ---
def plot_gauges(state):
    cols = st.columns(len(ASPECTS))
    latest_scores = {aspect: state["aspect_scores"][aspect][-1] if state["aspect_scores"][aspect] else 0 for aspect in ASPECTS}
    
    for idx, aspect in enumerate(ASPECTS):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest_scores[aspect],
            gauge={'axis': {'range': [0, 100]}},
            title={'text': aspect}
        ))
        fig.update_layout(height=250, margin=dict(t=20, b=20, l=10, r=10))
        cols[idx].plotly_chart(fig, use_container_width=True)

# --- Plot line chart for selected aspect ---
def plot_line_chart(state, selected_aspect):
    if state["timestamps"] and state["aspect_scores"][selected_aspect]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=state["timestamps"],
            y=state["aspect_scores"][selected_aspect],
            mode='lines+markers',
            name=selected_aspect
        ))
        fig.update_layout(
            title=f"{selected_aspect} Performance Over Time",
            xaxis_title="Time",
            yaxis_title="Score",
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Main App ---
st.set_page_config(page_title="Airline Service Performance Tracker", layout="wide")
st.title("✈ Airline Service Performance Tracker")

# Auto-refresh
st_autorefresh(interval=REFRESH_INTERVAL_SEC * 1000, key="datarefresh")

# Load state & data
state = load_state()
df = load_reviews()

# Process next review
process_next_review(state, df)

# Dropdown for aspect
selected_aspect = st.selectbox("Select Aspect to View Trend:", ASPECTS)

# Charts
plot_gauges(state)
plot_line_chart(state, selected_aspect)

# Info
st.markdown(f"**Last Processed Review Index:** {state['last_index'] + 1} / {len(df)}")
st.markdown(f"**Next Update In:** {REFRESH_INTERVAL_SEC} seconds")
