"""
simulator/01_ingest.py
───────────────────────
Phase 1 → Data Ingestion + Schema Validation

Run:
    python simulator/01_ingest.py
"""

import pandas as pd
from pathlib import Path

SCHEMA = {
    "flow_id":                  "string",
    "timestamp":                "string",
    "source_ip":                "string",
    "destination_ip":           "string",
    "source_port":              "int64",
    "destination_port":         "int64",
    "protocol":                 "category",
    "packet_count":             "int64",
    "byte_count":               "int64",
    "duration_seconds":         "float64",
    "connection_count":         "int64",
    "failed_connection_count":  "int64",
    "syn_count":                "int64",
    "rst_count":                "int64",
    "average_packet_size":      "float64",
    "label":                    "category",
    "scenario_type":            "category",
}

csv_path = Path("data/network_traffic.csv")
if not csv_path.exists():
    raise FileNotFoundError(
        "❌ data/network_traffic.csv not found.\n"
        "   Run: python simulator/generate_dataset.py first."
    )

print(f"📂 Loading {csv_path} …")
df = pd.read_csv(csv_path)
df = df.astype(SCHEMA)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ── Validations ────────────────────────────────────────────────────────────────
assert df["label"].isin(["NORMAL","SUSPICIOUS"]).all(),   "Invalid label found"
assert df["source_port"].between(0, 65535).all(),         "Invalid source port"
assert df["destination_port"].between(0, 65535).all(),    "Invalid destination port"
assert df["packet_count"].ge(0).all(),                    "Negative packet count"
assert df["byte_count"].ge(0).all(),                      "Negative byte count"
assert df["duration_seconds"].ge(0).all(),                "Negative duration"
assert df["flow_id"].nunique() == len(df),                "Duplicate flow IDs"

out = Path("data/network_traffic.parquet")
df.to_parquet(out, index=False)

n_susp = (df["label"] == "SUSPICIOUS").sum()
print(f"\n✅ Ingest complete → {out}")
print(f"   Shape          : {df.shape}")
print(f"   Normal flows   : {(df['label']=='NORMAL').sum():,}")
print(f"   Suspicious     : {n_susp:,}  ({n_susp/len(df)*100:.1f}%)")
print(f"   Date range     : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
print(f"   Null counts    :\n{df.isnull().sum()[df.isnull().sum()>0]}")
print(f"\n   dtypes:\n{df.dtypes}")
print("\n✅ Next → python simulator/traffic_simulator.py")