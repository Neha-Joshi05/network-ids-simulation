"""
ids/rule_engine.py
───────────────────
Phase 2 → Signature-Based Rule Engine

Applies 12 hand-crafted detection rules to each flow.
Each rule returns a risk score contribution (0–100).

Run:
    python ids/rule_engine.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")


# ── Rule definitions ─────────────────────────────────────────────────────────
UNUSUAL_PORTS = {4444, 6667, 31337, 1337, 12345, 54321, 9999, 7777}


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all rules to a DataFrame of flow records.
    Returns a DataFrame with per-rule scores + composite rule_score.
    """
    r = pd.DataFrame(index=df.index)

    # ── Rule 1: High connection rate (DoS indicator) ──────────────────────────
    r["rule_high_conn_rate"] = np.where(
        df["connection_count"] > 500, 90,
        np.where(df["connection_count"] > 200, 65,
        np.where(df["connection_count"] > 100, 40, 0))
    )

    # ── Rule 2: Repeated failed connections (brute-force / scan) ─────────────
    fail_ratio = df["failed_connection_count"] / (df["connection_count"] + 1)
    r["rule_failed_connections"] = np.where(
        (fail_ratio > 0.6) & (df["failed_connection_count"] > 20), 85,
        np.where((fail_ratio > 0.4) & (df["failed_connection_count"] > 10), 55,
        np.where(fail_ratio > 0.2, 25, 0))
    )

    # ── Rule 3: SYN flood pattern ─────────────────────────────────────────────
    syn_ratio = df["syn_count"] / (df["packet_count"] + 1)
    r["rule_syn_flood"] = np.where(
        (syn_ratio > 0.8) & (df["syn_count"] > 200), 95,
        np.where((syn_ratio > 0.6) & (df["syn_count"] > 50), 70,
        np.where(syn_ratio > 0.4, 30, 0))
    )

    # ── Rule 4: Port scan (multi-port probing) ────────────────────────────────
    r["rule_port_scan"] = np.where(
        (df["connection_count"] > 50) & (df["failed_connection_count"] > 40)
        & (df["packet_count"] < 10), 88,
        np.where(
        (df["connection_count"] > 20) & (fail_ratio > 0.5), 55, 0)
    )

    # ── Rule 5: Unusual destination port ─────────────────────────────────────
    r["rule_unusual_port"] = df["destination_port"].isin(UNUSUAL_PORTS).map(
        {True: 70, False: 0}
    )

    # ── Rule 6: High traffic volume (data exfiltration indicator) ─────────────
    r["rule_high_volume"] = np.where(
        df["byte_count"] > 50_000_000, 80,
        np.where(df["byte_count"] > 10_000_000, 50,
        np.where(df["byte_count"] > 1_000_000,  20, 0))
    )

    # ── Rule 7: Very high RST rate (scan/evasion) ─────────────────────────────
    rst_ratio = df["rst_count"] / (df["packet_count"] + 1)
    r["rule_rst_flood"] = np.where(
        rst_ratio > 0.5, 60,
        np.where(rst_ratio > 0.3, 35, 0)
    )

    # ── Rule 8: Tiny packets (evasion / covert channel) ───────────────────────
    r["rule_tiny_packets"] = np.where(
        (df["average_packet_size"] < 50) & (df["packet_count"] > 50), 55,
        np.where(df["average_packet_size"] < 40, 30, 0)
    )

    # ── Rule 9: Very long duration with high bytes (exfil/C2) ────────────────
    r["rule_long_duration_high_bytes"] = np.where(
        (df["duration_seconds"] > 300) & (df["byte_count"] > 5_000_000), 65,
        np.where(
        (df["duration_seconds"] > 120) & (df["byte_count"] > 1_000_000), 35, 0)
    )

    # ── Rule 10: SSH brute-force pattern ─────────────────────────────────────
    r["rule_ssh_bruteforce"] = np.where(
        (df["destination_port"] == 22) & (df["failed_connection_count"] > 15), 80,
        np.where(
        (df["destination_port"] == 22) & (df["failed_connection_count"] > 5), 45, 0)
    )

    # ── Rule 11: Extremely fast connection rate ───────────────────────────────
    conn_per_sec = df["connection_count"] / (df["duration_seconds"] + 0.001)
    r["rule_fast_conn_rate"] = np.where(
        conn_per_sec > 100, 85,
        np.where(conn_per_sec > 50, 55,
        np.where(conn_per_sec > 20, 25, 0))
    )

    # ── Rule 12: Suspicious ephemeral→ephemeral port pattern ─────────────────
    r["rule_ephemeral_ports"] = np.where(
        (df["source_port"] > 49151) & (df["destination_port"] > 49151)
        & (df["connection_count"] > 20), 45, 0
    )

    # ── Composite rule score (weighted max-pooling) ───────────────────────────
    rule_cols = [c for c in r.columns if c.startswith("rule_")]
    rule_matrix = r[rule_cols].values

    # Max of top-2 triggered rules + mean of the rest
    top2_mean  = np.sort(rule_matrix, axis=1)[:, -2:].mean(axis=1)
    rest_mean  = np.sort(rule_matrix, axis=1)[:, :-2].mean(axis=1)
    r["rule_score"] = (0.70 * top2_mean + 0.30 * rest_mean).clip(0, 100).round(2)

    # Alert flag
    r["rule_alert"] = r["rule_score"] >= 40

    return r


if __name__ == "__main__":
    print("📂 Loading network_traffic.parquet …")
    df = pd.read_parquet(DATA_DIR / "network_traffic.parquet")
    print(f"   {len(df):,} flows")

    print("🔧 Applying 12 detection rules …")
    results = apply_rules(df)

    out = DATA_DIR / "rule_scores.parquet"
    results.to_parquet(out, index=False)

    n_alerts = results["rule_alert"].sum()
    n_susp   = (df["label"] == "SUSPICIOUS").sum()

    print(f"\n   Rule alerts triggered : {n_alerts:,}  ({n_alerts/len(df)*100:.1f}%)")
    print(f"   True suspicious flows : {n_susp:,}  ({n_susp/len(df)*100:.1f}%)")
    print(f"\n   Rule score distribution:")
    print(results["rule_score"].describe().round(2).to_string())

    print(f"\n   Top triggered rules:")
    rule_cols = [c for c in results.columns if c.startswith("rule_") and c != "rule_score" and c != "rule_alert"]
    triggered = (results[rule_cols] > 0).sum().sort_values(ascending=False)
    print(triggered.to_string())

    print(f"\n✅ Rule engine done → {out}")
    print("   Next → python ids/anomaly_detector.py")