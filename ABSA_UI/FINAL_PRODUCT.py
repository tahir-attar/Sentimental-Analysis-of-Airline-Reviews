# FINAL_PRODUCT.py
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# NLP/model imports
import nltk
from nltk.tokenize import sent_tokenize
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ---------------- CONFIG ----------------
ASPECTS = ['food', 'seat', 'crew', 'staff', 'entertainment', 'comfort', 'flight', 'check-in', 'baggage']
INCOMING_DIR = "incoming_flights"  # folder containing flight CSVs
ABSA_CSV = "airline_reviews_with_aspect_sentiments.csv"
STATE_FILE = "state.json"
MODEL_DIR = "Model"
HISTORY_CAP = 1000
MAX_SEQ_LEN = 128
# ----------------------------------------

# ensure punkt
try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt", quiet=True)

# Streamlit setup
st.set_page_config(page_title="ABSA Flight Dashboard", layout="wide")
st.title("✈️ ABSA Flight Dashboard — Delhi Region (BOM → DEL)")

# ---------------- State helpers ----------------
def blank_state():
    return {
        "processed_files": [],       # filenames processed
        "next_file_idx": 0,          # index into sorted incoming list
        "history": {a: [] for a in ASPECTS},  # list of dicts: {t, pct, flight_no, route, flight_date, flight_time}
        "last_update": None
    }

def load_state():
    if Path(STATE_FILE).exists():
        try:
            with open(STATE_FILE, "r") as f:
                s = json.load(f)
            s.setdefault("processed_files", [])
            s.setdefault("next_file_idx", 0)
            s.setdefault("history", {a: [] for a in ASPECTS})
            for a in ASPECTS:
                s["history"].setdefault(a, [])
            s.setdefault("last_update", None)
            return s
        except Exception:
            pass
    return blank_state()

def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)

# ---------------- CSV / incoming files helpers ----------------
def list_incoming_files():
    Path(INCOMING_DIR).mkdir(parents=True, exist_ok=True)
    files = sorted([f for f in os.listdir(INCOMING_DIR) if f.lower().endswith(".csv")])
    return files

def parse_filename_meta(fname: str):
    base = Path(fname).stem
    parts = base.split("_")
    if len(parts) >= 3:
        flight_no = parts[0]
        route = parts[1]
        dt = parts[2]
        tt = parts[3] if len(parts) >= 4 else "0000"
        try:
            date_str = f"{dt[0:4]}-{dt[4:6]}-{dt[6:8]}"
            time_str = f"{tt[0:2]}:{tt[2:4]}"
        except Exception:
            date_str = dt
            time_str = tt
        return {"flight_no": flight_no, "route": route, "flight_date": date_str, "flight_time": time_str}
    return {"flight_no": "UNKNOWN", "route": "UNKNOWN", "flight_date": "", "flight_time": ""}

def ensure_absa_csv_with_meta(path):
    if not Path(path).exists():
        df = pd.DataFrame(columns=["Cleaned_Reviews", "Aspect_Sentiments", "Predicted_At",
                                   "Flight_No", "Route", "Flight_Date", "Flight_Time"])
        df.to_csv(path, index=False)

# ----------------- model loading -----------------
@st.cache_resource
def load_model_and_tokenizer(model_dir=MODEL_DIR):
    if not Path(model_dir).exists():
        st.error(f"Model folder not found: '{model_dir}'. Place your fine-tuned model in this folder.")
        st.stop()
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model_and_tokenizer()

# ---------- prediction helper ----------
def predict_sentiment_for_texts(texts: List[str]) -> List[str]:
    if not texts:
        return []
    enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
    logits = outputs.logits.cpu()
    preds = torch.argmax(logits, dim=1).tolist()
    labels = ['negative', 'positive']
    return [labels[p] if p < len(labels) else 'negative' for p in preds]

# ---------- ABSA pipeline ----------
def absa_pipeline_for_review(review_text: str) -> Dict[str, str]:
    out = {}
    if not isinstance(review_text, str) or not review_text.strip():
        return out
    sents = sent_tokenize(review_text)
    sents_lower = [s.lower() for s in sents]
    def aspect_match(aspect, sent_lower):
        a_norm = aspect.replace("-", " ").lower()
        return a_norm in sent_lower or aspect in sent_lower
    for aspect in ASPECTS:
        related = [s for s, sl in zip(sents, sents_lower) if aspect_match(aspect, sl)]
        if not related:
            related = [s for s, sl in zip(sents, sents_lower) if aspect.replace("-", " ") in sl or aspect.split("-")[0] in sl]
        if related:
            preds = predict_sentiment_for_texts(related)
            if preds:
                final = max(set(preds), key=preds.count)
                out[aspect] = final
    return out

# ---------- Append ABSA to CSV ----------
def append_to_absa_csv_with_meta(review_text: str, aspect_dict: Dict[str, str], meta: Dict[str,str]):
    ensure_absa_csv_with_meta(ABSA_CSV)
    row = {
        "Cleaned_Reviews": review_text,
        "Aspect_Sentiments": json.dumps(aspect_dict),
        "Predicted_At": datetime.utcnow().isoformat() + "Z",
        "Flight_No": meta.get("flight_no",""),
        "Route": meta.get("route",""),
        "Flight_Date": meta.get("flight_date",""),
        "Flight_Time": meta.get("flight_time","")
    }
    df_row = pd.DataFrame([row])
    header = not Path(ABSA_CSV).exists()
    df_row.to_csv(ABSA_CSV, mode="a", header=header, index=False)

# ---------- Aggregate flight-level aspect % ----------
def aggregate_flight_aspect_pct(aspect_results: List[Dict[str,str]]) -> Dict[str,float]:
    counts = {a: {"pos":0, "neg":0} for a in ASPECTS}
    for res in aspect_results:
        for a, val in res.items():
            if val == "positive":
                counts[a]["pos"] += 1
            elif val == "negative":
                counts[a]["neg"] += 1
    pct = {}
    for a in ASPECTS:
        pos = counts[a]["pos"]
        neg = counts[a]["neg"]
        pct[a] = (pos / (pos + neg) * 100.0) if (pos + neg) > 0 else 0.0
    return pct

# ---------- UI helpers ----------
def latest_and_prev(history_list):
    if not history_list:
        return 0.0, 0.0
    curr = history_list[-1]["pct"]
    prev = history_list[-2]["pct"] if len(history_list) > 1 else history_list[-1]["pct"]
    return curr, prev

def aspect_gauge(aspect, curr_pct, prev_pct):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=curr_pct,
        delta={"reference": prev_pct, "relative": False,
               "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
        title={"text": aspect.capitalize()},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "royalblue"}}
    ))
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=260)
    return fig

# ---------- Safe rerun helper ----------
def safe_rerun():
    if 'rerun_trigger' not in st.session_state:
        st.session_state['rerun_trigger'] = False
    st.session_state['rerun_trigger'] = not st.session_state['rerun_trigger']

# ---------------- Main flow ----------------
state = load_state()
all_files = list_incoming_files()
ensure_absa_csv_with_meta(ABSA_CSV)

# Sidebar
st.sidebar.subheader("Incoming Flights")
for i, f in enumerate(all_files):
    marker = "✅" if f in state["processed_files"] else ("➡" if i == state.get("next_file_idx",0) else "•")
    st.sidebar.text(f"{marker} {f}")

next_idx = state.get("next_file_idx", 0)
if next_idx >= len(all_files):
    st.info("No unprocessed flight files remaining.")
else:
    next_file = all_files[next_idx]
    meta = parse_filename_meta(next_file)
    st.subheader(f"Next flight: {meta['flight_no']} — {meta['route']} — {meta['flight_date']} {meta['flight_time']}")
    col1, col2 = st.columns(2)
    with col1:
        process_button = st.button("▶ Process This Flight")
    with col2:
        skip_button = st.button("⏭ Skip This Flight")
    if skip_button:
        state["processed_files"].append(next_file)
        state["next_file_idx"] = next_idx + 1
        state["last_update"] = datetime.utcnow().isoformat() + "Z"
        save_state(state)
        safe_rerun()
    if process_button:
        file_path = os.path.join(INCOMING_DIR, next_file)
        df_in = pd.read_csv(file_path)
        possible_cols = ["Cleaned_Reviews","review","Review","cleaned_review"]
        found_col = next((c for c in possible_cols if c in df_in.columns), None)
        if not found_col:
            st.error(f"No review column found in {next_file}.")
        else:
            reviews = df_in[found_col].fillna("").astype(str).tolist()
            n = len(reviews)
            progress_bar = st.progress(0)
            status = st.empty()
            aspect_results_for_flight = []
            start_time = time.time()
            for i, review in enumerate(reviews, start=1):
                aspect_map = absa_pipeline_for_review(review)
                append_to_absa_csv_with_meta(review, aspect_map, meta)
                aspect_results_for_flight.append(aspect_map)
                progress_bar.progress(i / n)
                elapsed = time.time() - start_time
                remaining = (elapsed / i) * (n - i) if i > 0 else 0
                status.text(f"Processed {i}/{n} — ETA ~{int(remaining)}s")
            
            flight_pct = aggregate_flight_aspect_pct(aspect_results_for_flight)
            ts = datetime.utcnow().isoformat() + "Z"
            for a in ASPECTS:
                state["history"].setdefault(a, [])
                state["history"][a].append({
                    "t": ts,
                    "pct": round(flight_pct.get(a, 0.0), 2),
                    "flight_no": meta["flight_no"],
                    "route": meta["route"],
                    "flight_date": meta["flight_date"],
                    "flight_time": meta["flight_time"]
                })
                if len(state["history"][a]) > HISTORY_CAP:
                    state["history"][a] = state["history"][a][-HISTORY_CAP:]
            
            state["processed_files"].append(next_file)
            state["next_file_idx"] = next_idx + 1
            state["last_update"] = ts
            save_state(state)
            
            progress_bar.empty()
            status.success(f"Finished processing {next_file}.")
            safe_rerun()

# ---------------- Dashboard ----------------
st.markdown("---")
st.subheader("Aspect Trend Over Flights")
selected_aspect = st.selectbox("Select Aspect", ASPECTS, index=0)
hist = state.get("history", {}).get(selected_aspect, [])
if hist:
    df_hist = pd.DataFrame(hist)
    df_hist["time_idx"] = pd.to_datetime(df_hist["flight_date"] + " " + df_hist["flight_time"], errors='coerce')
    df_hist["hover"] = df_hist.apply(lambda r: f"{r['flight_no']}, {r['route']}, {r['flight_date']} {r['flight_time']}", axis=1)
    fig = go.Figure(go.Scatter(
        x=df_hist["time_idx"],
        y=df_hist["pct"],
        mode="lines+markers",
        marker=dict(size=10),
        hovertext=df_hist["hover"],
        hoverinfo="text+y"
    ))
    fig.update_layout(
        title=f"{selected_aspect.capitalize()} — Positive % per Flight",
        xaxis_title="Flight Time",
        yaxis_title="Positive %",
        yaxis=dict(range=[0,100]),
        height=350,
        margin=dict(t=40,b=10,l=10,r=10)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No flight history yet.")

st.subheader("Aspect Gauges (Latest Flight vs Previous)")
cols_per_row = 3
row_cols = None
for i, a in enumerate(ASPECTS):
    if i % cols_per_row == 0:
        row_cols = st.columns(cols_per_row)
    history_a = state.get("history", {}).get(a, [])
    curr, prev = latest_and_prev(history_a)
    fig = aspect_gauge(a, curr, prev)
    row_cols[i % cols_per_row].plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
c1, c2 = st.columns(2)
with c1:
    st.caption(f"Incoming folder: `{INCOMING_DIR}`")
    st.caption(f"Processed files: {len(state.get('processed_files',[]))} / {len(all_files)}")
with c2:
    st.caption(f"ABSA CSV: `{ABSA_CSV}`")
    st.caption(f"Last update (UTC): {state.get('last_update', '—')}")
