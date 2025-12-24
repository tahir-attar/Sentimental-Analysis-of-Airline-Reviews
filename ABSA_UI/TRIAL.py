# absa_live_app.py
import os
import json
import time
import ast
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# NLP/model imports
import nltk
from nltk.tokenize import sent_tokenize
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ---------------- CONFIG ----------------
ASPECTS = ['food', 'seat', 'crew', 'staff', 'entertainment', 'comfort', 'flight', 'check-in', 'baggage']
CLEANED_CSV = "cleaned_airlines_reviews.csv"                       # source reviews (one-by-one)
ABSA_CSV = "airline_reviews_with_aspect_sentiments.csv"            # destination where we append ABSA results
STATE_FILE = "state.json"                                         # tracks last processed index + history
MODEL_DIR = "Model"                                                # folder with tokenizer + model
REFRESH_DEFAULT_SEC = 5
BATCH_DEFAULT = 1
HISTORY_CAP = 500
MAX_SEQ_LEN = 128
# ----------------------------------------

# ensure punkt for sent_tokenize
try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt", quiet=True)

# --------- Streamlit page setup ----------
st.set_page_config(page_title="ABSA Live App", layout="wide")
st.title("✈️ ABSA Live Pipeline + Dashboard")

# Sidebar controls
refresh_sec = st.sidebar.slider("Auto-refresh interval (seconds)", 2, 60, value=REFRESH_DEFAULT_SEC)
batch_size = st.sidebar.slider("Process reviews per refresh", 1, 10, value=BATCH_DEFAULT)
st.sidebar.caption("Pipeline: cleaned_reviews → ABSA model → append ABSA CSV → dashboard update")

st_autorefresh(interval=refresh_sec * 1000, key="live_tick")  # auto refresh

# ----------------- helpers: state -----------------
def blank_state():
    return {
        "file_path": None,
        "file_fingerprint": "",
        "last_index": -1,
        "prefix_hash": "",
        "counts": {a: {"pos": 0, "neg": 0} for a in ASPECTS},
        "history": {a: [] for a in ASPECTS},   # list of {"t":iso, "pct": float}
        "last_update": None
    }

def load_state():
    if Path(STATE_FILE).exists():
        try:
            with open(STATE_FILE, "r") as f:
                s = json.load(f)
            # ensure keys
            s.setdefault("counts", {a: {"pos": 0, "neg": 0} for a in ASPECTS})
            s.setdefault("history", {a: [] for a in ASPECTS})
            for a in ASPECTS:
                s["counts"].setdefault(a, {"pos": 0, "neg": 0})
                s["history"].setdefault(a, [])
            return s
        except Exception:
            pass
    return blank_state()

def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)

# ----------------- helpers: CSV & fingerprint -----------------
def load_cleaned_csv(path):
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    return df.reset_index(drop=True)

def ensure_absa_csv(path):
    # create ABSA CSV with required columns if not exists
    if not Path(path).exists():
        df = pd.DataFrame(columns=["Cleaned_Reviews", "Aspect_Sentiments", "Predicted_At"])
        df.to_csv(path, index=False)

def file_fingerprint(path):
    p = Path(path)
    if not p.exists():
        return ""
    stt = p.stat()
    size = stt.st_size
    mtime = int(stt.st_mtime)
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return f"{size}-{mtime}-{h.hexdigest()}"

def row_key(df, idx):
    r = df.iloc[idx]
    text = str(r.get("Cleaned_Reviews", "")).strip()
    return text

def compute_prefix_hash(df, upto_idx):
    h = hashlib.md5()
    if upto_idx >= 0:
        for i in range(0, upto_idx + 1):
            try:
                k = row_key(df, i)
            except Exception:
                k = ""
            h.update(k.encode("utf-8", errors="ignore"))
    return h.hexdigest()

# ----------------- model loading -----------------
@st.cache_resource
def load_model_and_tokenizer(model_dir=MODEL_DIR):
    if not Path(model_dir).exists():
        st.error(f"Model folder not found: '{model_dir}'. Place your fine-tuned model in this folder.")
        st.stop()
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    # attempt to use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model_and_tokenizer()

# ---------- prediction helper ----------
def predict_sentiment_for_texts(texts):
    """
    texts: list[str], returns list['positive' or 'negative']
    """
    if not texts:
        return []
    # tokenize batch
    enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
    logits = outputs.logits.cpu()
    preds = torch.argmax(logits, dim=1).tolist()
    # labels assumed ['negative','positive']
    labels = ['negative', 'positive']
    return [labels[p] if p < len(labels) else 'negative' for p in preds]

# ---------- ABSA pipeline for one review ----------
def absa_pipeline_for_review(review_text):
    """
    - Sentence-tokenize review_text
    - For each aspect, find sentences containing the aspect keyword (simple presence)
    - If none found, ignore aspect for that review
    - If sentences found, predict sentiment on those sentences and use majority vote
    Returns dict: aspect -> 'positive'/'negative'
    """
    out = {}
    if not isinstance(review_text, str) or not review_text.strip():
        return out
    sents = sent_tokenize(review_text)
    sents_lower = [s.lower() for s in sents]
    # normalize aspects keys for matching: handle 'check-in' -> check in
    def aspect_match(aspect, sent_lower):
        a_norm = aspect.replace("-", " ").lower()
        return a_norm in sent_lower or aspect in sent_lower

    for aspect in ASPECTS:
        related = [s for s, sl in zip(sents, sents_lower) if aspect_match(aspect, sl)]
        if not related:
            # also try single-word tokens (e.g., 'seat' might be 'seats')
            related = [s for s, sl in zip(sents, sents_lower) if aspect.replace("-", " ") in sl or aspect.split("-")[0] in sl]
        if related:
            preds = predict_sentiment_for_texts(related)
            # majority vote
            if preds:
                final = max(set(preds), key=preds.count)
                out[aspect] = final
    return out

# ---------- Append ABSA to ABSA CSV ----------
def append_to_absa_csv(review_text, aspect_dict):
    ensure_absa_csv(ABSA_CSV)
    row = {
        "Cleaned_Reviews": review_text,
        "Aspect_Sentiments": json.dumps(aspect_dict),
        "Predicted_At": datetime.utcnow().isoformat() + "Z"
    }
    df_row = pd.DataFrame([row])
    df_row.to_csv(ABSA_CSV, mode="a", header=not Path(ABSA_CSV).exists(), index=False)

# ---------- progress ingestion ----------
def ingest_next_batch(state, cleaned_df, batch_n):
    start = state["last_index"] + 1
    if start >= len(cleaned_df):
        return False
    end = min(len(cleaned_df), start + batch_n)
    for idx in range(start, end):
        review_text = str(cleaned_df.iloc[idx].get("Cleaned_Reviews", "")).strip()
        # get ABSA predictions using model
        aspect_map = absa_pipeline_for_review(review_text)
        # append to absa csv
        append_to_absa_csv(review_text, aspect_map)
        # update counts & history in state
        ts = datetime.utcnow().isoformat() + "Z"
        for a in ASPECTS:
            if a in aspect_map:
                if aspect_map[a] == "positive":
                    state["counts"][a]["pos"] += 1
                elif aspect_map[a] == "negative":
                    state["counts"][a]["neg"] += 1
            # compute pct snapshot
            pos = state["counts"][a]["pos"]
            neg = state["counts"][a]["neg"]
            pct = (pos / (pos + neg) * 100.0) if (pos + neg) > 0 else 0.0
            state["history"][a].append({"t": ts, "pct": round(pct, 2)})
            if len(state["history"][a]) > HISTORY_CAP:
                state["history"][a] = state["history"][a][-HISTORY_CAP:]
        state["last_index"] = idx
        state["last_update"] = ts
    return True

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

# ---------------- Main flow ----------------
state = load_state()

cleaned_df = load_cleaned_csv(CLEANED_CSV)
if cleaned_df is None:
    st.error(f"Source cleaned CSV not found: '{CLEANED_CSV}'. Place it in the app folder.")
    st.stop()

# Prepare ABSA CSV if missing
ensure_absa_csv(ABSA_CSV)

# Continuity check on cleaned CSV (hot-swap)
current_fp = file_fingerprint(CLEANED_CSV)
prefix_hash_now = compute_prefix_hash(cleaned_df, state["last_index"])
if state.get("file_fingerprint", "") != current_fp:
    if state.get("last_index", -1) >= 0:
        # if prefix changed, reset (safe)
        if state.get("prefix_hash", "") != prefix_hash_now:
            st.warning("cleaned CSV changed and previously-processed prefix doesn't match: resetting progress to start of cleaned CSV.")
            state = blank_state()
    state["file_fingerprint"] = current_fp
    state["prefix_hash"] = compute_prefix_hash(cleaned_df, state["last_index"])
    save_state(state)

# Ingest next batch (runs model and appends to ABSA CSV)
did_ingest = ingest_next_batch(state, cleaned_df, batch_size)
# update prefix hash and save
state["prefix_hash"] = compute_prefix_hash(cleaned_df, state["last_index"])
save_state(state)

# ========== Dashboard UI ==========
st.subheader("Aspect Trend Over Time")
sel = st.selectbox("Select aspect", ASPECTS, index=0)
hist = state["history"].get(sel, [])
if hist:
    trend_df = pd.DataFrame({"Time": pd.to_datetime([h["t"] for h in hist]), "Positive %": [h["pct"] for h in hist]}).set_index("Time")
    st.line_chart(trend_df, use_container_width=True)
else:
    st.info("Waiting for first predictions...")

st.subheader("Aspect Gauges (Positive % with delta)")
cols_per_row = 3
row_cols = None
for i, a in enumerate(ASPECTS):
    if i % cols_per_row == 0:
        row_cols = st.columns(cols_per_row)
    curr, prev = latest_and_prev(state["history"].get(a, []))
    fig = aspect_gauge(a, curr, prev)
    row_cols[i % cols_per_row].plotly_chart(fig, use_container_width=True)

# Footer info
left, right = st.columns(2)
with left:
    st.caption(f"Source cleaned CSV: **{CLEANED_CSV}**")
    st.caption(f"ABSA CSV: **{ABSA_CSV}**")
with right:
    st.caption(f"Processed reviews: **{state.get('last_index', -1) + 1} / {len(cleaned_df)}**")
    st.caption(f"Last update (UTC): **{state.get('last_update','—')}**")

if not did_ingest:
    st.info("No new reviews to process. Add rows to the cleaned CSV or replace it with a new one to continue.")
