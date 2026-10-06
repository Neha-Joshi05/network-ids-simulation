"""
ids/feature_extractor.py
─────────────────────────
Phase 2 → Feature Engineering for IDS

Run:
    python ids/feature_extractor.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

UNUSUAL_PORTS = {4444, 6667, 31337, 1337, 12345, 54321, 9999, 7777, 8888, 2222}
COMMON_PORTS  = {80, 443, 53, 22, 25, 110, 143, 3306, 5432, 8080, 8443, 21, 23}


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame()

    # Raw numerics
    f["packet_count"]            = df["packet_count"]
    f["byte_count"]              = df["byte_count"]
    f["duration_seconds"]        = df["duration_seconds"]
    f["connection_count"]        = df["connection_count"]
    f["failed_connection_count"] = df["failed_connection_count"]
    f["syn_count"]               = df["syn_count"]
    f["rst_count"]               = df["rst_count"]
    f["average_packet_size"]     = df["average_packet_size"]
    f["source_port"]             = df["source_port"]
    f["destination_port"]        = df["destination_port"]

    # Derived ratios
    f["failed_conn_ratio"]    = (df["failed_connection_count"] / (df["connection_count"] + 1)).clip(0, 1)
    f["syn_ratio"]            = (df["syn_count"] / (df["packet_count"] + 1)).clip(0, 1)
    f["rst_ratio"]            = (df["rst_count"] / (df["packet_count"] + 1)).clip(0, 1)
    f["bytes_per_second"]     = (df["byte_count"] / (df["duration_seconds"] + 0.001)).clip(0)
    f["packets_per_second"]   = (df["packet_count"] / (df["duration_seconds"] + 0.001)).clip(0)
    f["bytes_per_connection"] = (df["byte_count"] / (df["connection_count"] + 1))
    f["log_byte_count"]       = np.log1p(df["byte_count"])
    f["log_packet_count"]     = np.log1p(df["packet_count"])
    f["log_connection_count"] = np.log1p(df["connection_count"])

    # Port flags
    f["is_unusual_port"]       = df["destination_port"].isin(UNUSUAL_PORTS).astype(int)
    f["is_common_port"]        = df["destination_port"].isin(COMMON_PORTS).astype(int)
    f["is_ephemeral_src_port"] = (df["source_port"] > 49151).astype(int)
    f["port_diff"]             = (df["destination_port"] - df["source_port"]).abs()

    # Protocol flags
    f["is_tcp"]  = (df["protocol"] == "TCP").astype(int)
    f["is_udp"]  = (df["protocol"] == "UDP").astype(int)
    f["is_icmp"] = (df["protocol"] == "ICMP").astype(int)
    f["is_dns"]  = (df["protocol"] == "DNS").astype(int)
    f["is_ssh"]  = (df["protocol"] == "SSH").astype(int)

    # Risk composite flags
    f["high_conn_rate"]  = (df["connection_count"] > 100).astype(int)
    f["high_fail_rate"]  = (f["failed_conn_ratio"] > 0.4).astype(int)
    f["high_syn_rate"]   = (f["syn_ratio"] > 0.6).astype(int)
    f["burst_traffic"]   = (f["bytes_per_second"] > 1_000_000).astype(int)
    f["risk_flag_sum"]   = (f["high_conn_rate"] + f["high_fail_rate"] +
                             f["high_syn_rate"]  + f["burst_traffic"] +
                             f["is_unusual_port"])
    return f


if __name__ == "__main__":
    print("📂 Loading network_traffic.parquet …")
    df = pd.read_parquet(DATA_DIR / "network_traffic.parquet")
    print(f"   {len(df):,} flow records")

    print("🔧 Extracting features …")
    features = extract_features(df)
    feature_cols = list(features.columns)
    print(f"   {len(feature_cols)} features extracted")

    joblib.dump(feature_cols, MODEL_DIR / "feature_list.joblib")
    features.to_parquet(DATA_DIR / "features.parquet", index=False)

    print(f"   ✅ models/feature_list.joblib")
    print(f"   ✅ data/features.parquet")
    print(f"\n   Features: {feature_cols}")
    print("\n✅ Done! Next → python ids/rule_engine.py")