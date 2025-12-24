# feeder.py
import os
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
import pandas as pd

# ========= CONFIG =========
CSV_PATH = "airline_reviews_with_aspect_sentiments.csv"  # replace anytime
STATE_PATH = "state.json"                                # persistent state
FEED_INTERVAL_SEC = 6                                    # 1 review every N secs
# If your CSV columns aren't known, leave ASPECT_COLUMNS = None (auto-detects numerics)
ASPECT_COLUMNS = None  # e.g. ["food", "seat", "crew", "staff", "entertainment", "comfort"]
# =========================

def file_fingerprint(path: str) -> str:
    """Robust fingerprint: size + mtime + first 64KB hash (fast)."""
    p = Path(path)
    if not p.exists():
        return ""
    size = p.stat().st_size
    mtime = int(p.stat().st_mtime)
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return f"{size}-{mtime}-{h.hexdigest()}"

def load_state() -> dict:
    if Path(STATE_PATH).exists():
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    return {
        "csv_fingerprint": "",
        "last_index": -1,
        "aspects": [],
        "latest": {},                  # {aspect: value}
        "history": {},                 # {aspect: [{"t": iso, "v": value}, ...]}
        "last_update": None
    }

def save_state(state: dict):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

def coerce_value(val):
    """Convert common sentiment formats to 0..100 float."""
    # numeric already?
    try:
        num = float(val)
        # If it's in 0..1 range, treat as probability; else assume 0..100
        if 0.0 <= num <= 1.0:
            return round(num * 100.0, 2)
        return round(num, 2)
    except Exception:
        pass
    # string labels
    s = str(val).strip().lower()
    if s in {"pos", "positive", "good", "satisfied", "yes"}:
        return 100.0
    if s in {"neg", "negative", "bad", "unsatisfied", "no"}:
        return 0.0
    if s in {"neutral", "mixed"}:
        return 50.0
    # unknown → ignore by returning None
    return None

def detect_aspects(df: pd.DataFrame) -> list:
    """If ASPECT_COLUMNS not provided, infer numeric-like aspect columns."""
    if ASPECT_COLUMNS:
        return ASPECT_COLUMNS
    # Heuristic: numeric columns OR columns containing known aspect keywords
    keywords = ["food", "seat", "crew", "staff", "service", "entertainment",
                "comfort", "clean", "baggage", "check", "value", "flight"]
    candidates = []
    for col in df.columns:
        low = col.lower()
        if any(k in low for k in keywords):
            candidates.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            candidates.append(col)
    # De-dup & keep order
    seen = set()
    result = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result

def load_csv(csv_path: str) -> pd.DataFrame:
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Ensure deterministic ordering if no time column
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    elif "date" in df.columns:
        df = df.sort_values("date")
    df = df.reset_index(drop=True)
    return df

def append_point(state: dict, timestamp_iso: str, aspect_values: dict):
    # init structures if new aspects appear
    for a in aspect_values.keys():
        state["history"].setdefault(a, [])
    # update latest + history
    for a, v in aspect_values.items():
        if v is None:
            continue
        state["latest"][a] = float(v)
        state["history"][a].append({"t": timestamp_iso, "v": float(v)})
        # keep last 100 points per aspect to cap size
        if len(state["history"][a]) > 100:
            state["history"][a] = state["history"][a][-100:]

def process_next_row(state: dict, df: pd.DataFrame):
    next_idx = state["last_index"] + 1
    if next_idx >= len(df):
        return False  # nothing to do
    row = df.iloc[next_idx]
    timestamp_iso = datetime.utcnow().isoformat() + "Z"

    # Determine aspects list for this CSV
    if not state["aspects"]:
        state["aspects"] = detect_aspects(df)

    # Extract + coerce values
    aspect_values = {}
    for a in state["aspects"]:
        if a in df.columns:
            v = coerce_value(row[a])
            aspect_values[a] = v
    append_point(state, timestamp_iso, aspect_values)

    state["last_index"] = next_idx
    state["last_update"] = timestamp_iso
    return True

def main():
    print("🔁 Live feeder starting…")
    state = load_state()

    while True:
        # Reload CSV if replaced
        fp = file_fingerprint(CSV_PATH)
        try:
            df = load_csv(CSV_PATH)
        except FileNotFoundError:
            print("❗ Waiting for CSV…")
            time.sleep(FEED_INTERVAL_SEC)
            continue

        if state["csv_fingerprint"] != fp:
            print("📄 CSV changed or first run — (re)loading.")
            state["csv_fingerprint"] = fp
            # If CSV replaced, reset pointer but KEEP history
            state["last_index"] = -1
            # Also refresh aspects list from new CSV
            state["aspects"] = detect_aspects(df)
            save_state(state)

        did = process_next_row(state, df)
        if did:
            save_state(state)
            print(f"✅ Added row #{state['last_index']+1}/{len(df)} | aspects: {', '.join(state['aspects'])}")
        else:
            # No new rows; wait a bit, then check again (maybe CSV will be replaced)
            print("⏸ No new rows; idling.")
        time.sleep(FEED_INTERVAL_SEC)

if __name__ == "__main__":
    main()
