"""
ids/anomaly_detector.py
────────────────────────
Phase 2 → Anomaly-Based Detection (Isolation Forest)

Trains an unsupervised Isolation Forest on NORMAL flows,
then scores all flows for anomaly level.

Run:
    python ids/anomaly_detector.py
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATA_DIR   = Path("data")
MODEL_DIR  = Path("models")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

ANOMALY_FEATURES = [
    "packet_count", "byte_count", "duration_seconds",
    "connection_count", "failed_connection_count",
    "syn_count", "rst_count", "average_packet_size",
    "failed_conn_ratio", "syn_ratio", "rst_ratio",
    "bytes_per_second", "packets_per_second",
    "bytes_per_connection", "log_byte_count",
    "log_packet_count", "log_connection_count",
    "risk_flag_sum",
]

if __name__ == "__main__":
    print("📂 Loading features …")
    features = pd.read_parquet(DATA_DIR / "features.parquet")
    df_raw   = pd.read_parquet(DATA_DIR / "network_traffic.parquet")
    print(f"   {len(features):,} flows")

    # ── Filter available features ─────────────────────────────────────────────
    available = [f for f in ANOMALY_FEATURES if f in features.columns]
    X = features[available].fillna(0)

    # ── Train on NORMAL flows only (unsupervised) ─────────────────────────────
    normal_mask = df_raw["label"] == "NORMAL"
    X_normal    = X[normal_mask]
    print(f"   Training on {X_normal.shape[0]:,} NORMAL flows …")

    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    X_all_scaled    = scaler.transform(X)

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.15,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_normal_scaled)

    # ── Score all flows ───────────────────────────────────────────────────────
    raw_scores     = iso.decision_function(X_all_scaled)   # higher = more normal
    anomaly_scores = (1 - (raw_scores - raw_scores.min()) /
                      (raw_scores.max() - raw_scores.min() + 1e-9)) * 100

    results = pd.DataFrame({
        "anomaly_score": anomaly_scores.round(2),
        "is_anomaly":    iso.predict(X_all_scaled) == -1,
    })

    # ── Save ──────────────────────────────────────────────────────────────────
    results.to_parquet(DATA_DIR / "anomaly_scores.parquet", index=False)
    joblib.dump(iso,    MODEL_DIR / "isolation_forest.joblib")
    joblib.dump(scaler, MODEL_DIR / "anomaly_scaler.joblib")
    joblib.dump(available, MODEL_DIR / "anomaly_features.joblib")

    n_anomalies = results["is_anomaly"].sum()
    n_susp      = (df_raw["label"] == "SUSPICIOUS").sum()

    print(f"\n   Anomalies detected : {n_anomalies:,}  ({n_anomalies/len(results)*100:.1f}%)")
    print(f"   True suspicious    : {n_susp:,}  ({n_susp/len(df_raw)*100:.1f}%)")
    print(f"\n   Anomaly score stats:")
    print(results["anomaly_score"].describe().round(2).to_string())

    # ── Chart: Anomaly score distribution ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Isolation Forest — Anomaly Detection", fontsize=13, fontweight="bold")

    sns.histplot(data=pd.DataFrame({
        "score": anomaly_scores,
        "label": df_raw["label"].values
    }), x="score", hue="label", bins=60, ax=axes[0],
        palette={"NORMAL":"#2ecc71","SUSPICIOUS":"#e74c3c"})
    axes[0].set_title("Anomaly Score Distribution")
    axes[0].set_xlabel("Anomaly Score (0=Normal, 100=Anomalous)")

    axes[1].scatter(
        range(min(2000, len(results))),
        anomaly_scores[:2000],
        c=["#e74c3c" if a else "#2ecc71" for a in results["is_anomaly"][:2000]],
        alpha=0.4, s=8
    )
    axes[1].axhline(50, color="orange", linestyle="--", lw=1.5, label="Threshold=50")
    axes[1].set_title("Anomaly Scores (first 2000 flows)")
    axes[1].set_xlabel("Flow Index")
    axes[1].set_ylabel("Anomaly Score")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_anomaly_detection.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n   ✅ outputs/01_anomaly_detection.png")

    print(f"   ✅ models/isolation_forest.joblib")
    print(f"   ✅ models/anomaly_scaler.joblib")
    print(f"   ✅ data/anomaly_scores.parquet")
    print("\n✅ Anomaly detection complete!")
    print("   Next → python ml/train_classifier.py")