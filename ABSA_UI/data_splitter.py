import pandas as pd
import os
from datetime import datetime, timedelta

# === SETTINGS ===
INPUT_FILE = "cleaned_airlines_reviews.csv"
OUTPUT_DIR = "incoming_flights"
REVIEWS_PER_FILE = 150
START_FLIGHT_NO = 301
TOTAL_FILES = 54

# === MAKE OUTPUT DIR ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(INPUT_FILE).dropna(subset=["Cleaned_Reviews"]).reset_index(drop=True)
print(f"Total reviews loaded: {len(df)}")

# === SPLITTING ===
start_time = datetime(2025, 10, 1, 6, 0)  # first flight 2025-10-01 06:00
time_gap = timedelta(hours=4)

for i in range(TOTAL_FILES):
    start_idx = i * REVIEWS_PER_FILE
    end_idx = min((i + 1) * REVIEWS_PER_FILE, len(df))
    
    chunk = df.iloc[start_idx:end_idx]
    
    if chunk.empty:
        break
    
    flight_no = START_FLIGHT_NO + i
    flight_code = f"AI{flight_no}"
    
    flight_time = start_time + i * time_gap
    timestamp = flight_time.strftime("%Y%m%d_%H%M")
    
    filename = f"{flight_code}_BOM-DEL_{timestamp}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    chunk.to_csv(filepath, index=False)
    print(f"Saved: {filename} ({len(chunk)} reviews)")

print("✅ Splitting complete. Files are saved in 'incoming_flights/' folder.")
