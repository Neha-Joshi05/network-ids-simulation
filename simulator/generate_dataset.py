"""
simulator/generate_dataset.py
──────────────────────────────
Generates a synthetic network traffic dataset for IDS simulation.

All IPs use RFC 5737 documentation ranges (safe, non-routable):
  192.0.2.0/24   (TEST-NET-1)
  198.51.100.0/24 (TEST-NET-2)
  203.0.113.0/24  (TEST-NET-3)

NO real network packets are generated — only DATA RECORDS.

Run:
    python simulator/generate_dataset.py

Output:
    data/network_traffic.csv
    data/network_traffic.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
SEED       = 42
N_TOTAL    = 6000
OUTPUT_DIR = Path("data")

np.random.seed(SEED)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Safe documentation IP pools (RFC 5737) ────────────────────────────────────
INTERNAL_IPS = [f"192.0.2.{i}"    for i in range(1, 50)]
EXTERNAL_IPS = [f"198.51.100.{i}" for i in range(1, 50)]
SERVER_IPS   = [f"203.0.113.{i}"  for i in range(1, 20)]
ALL_SRC_IPS  = INTERNAL_IPS + EXTERNAL_IPS
ALL_DST_IPS  = SERVER_IPS   + INTERNAL_IPS

PROTOCOLS    = ["TCP", "UDP", "ICMP", "DNS", "HTTP", "HTTPS", "SSH", "FTP"]
NORMAL_PORTS = [80, 443, 53, 22, 25, 110, 143, 3306, 5432, 8080, 8443]
UNUSUAL_PORTS= [4444, 6667, 31337, 1337, 12345, 54321, 9999, 7777]

# ── Scenario definitions ──────────────────────────────────────────────────────
SCENARIOS = {
    "NORMAL_WEB":                  0.25,
    "NORMAL_DNS":                  0.15,
    "NORMAL_SSH":                  0.10,
    "NORMAL_EMAIL":                0.08,
    "NORMAL_DATABASE":             0.07,
    "HIGH_CONNECTION_RATE":        0.10,
    "REPEATED_FAILED_CONNECTIONS": 0.08,
    "MULTI_PORT_PROBING_PATTERN":  0.07,
    "SYN_HEAVY_PATTERN":           0.05,
    "UNUSUAL_PORT_ACTIVITY":       0.03,
    "HIGH_TRAFFIC_VOLUME":         0.02,
}

scenario_names = list(SCENARIOS.keys())
scenario_probs = list(SCENARIOS.values())
n_suspicious   = int(N_TOTAL * 0.35)
n_normal       = N_TOTAL - n_suspicious


def make_normal_web(n):
    return dict(
        protocol       = np.random.choice(["HTTP","HTTPS"], n, p=[0.3,0.7]),
        source_port    = np.random.randint(49152, 65535, n),
        destination_port = np.random.choice([80, 443], n),
        packet_count   = np.random.randint(5, 50, n),
        byte_count     = np.random.randint(1000, 80000, n),
        duration_seconds = np.random.uniform(0.5, 10, n),
        connection_count = np.random.randint(1, 5, n),
        failed_connection_count = np.zeros(n, dtype=int),
        syn_count      = np.random.randint(1, 4, n),
        rst_count      = np.zeros(n, dtype=int),
        label          = ["NORMAL"] * n,
        scenario_type  = ["NORMAL_WEB"] * n,
    )


def make_normal_dns(n):
    return dict(
        protocol       = ["DNS"] * n,
        source_port    = np.random.randint(49152, 65535, n),
        destination_port = [53] * n,
        packet_count   = np.random.randint(2, 10, n),
        byte_count     = np.random.randint(100, 1000, n),
        duration_seconds = np.random.uniform(0.01, 0.5, n),
        connection_count = np.random.randint(1, 3, n),
        failed_connection_count = np.zeros(n, dtype=int),
        syn_count      = np.zeros(n, dtype=int),
        rst_count      = np.zeros(n, dtype=int),
        label          = ["NORMAL"] * n,
        scenario_type  = ["NORMAL_DNS"] * n,
    )


def make_normal_ssh(n):
    return dict(
        protocol       = ["SSH"] * n,
        source_port    = np.random.randint(49152, 65535, n),
        destination_port = [22] * n,
        packet_count   = np.random.randint(20, 200, n),
        byte_count     = np.random.randint(5000, 200000, n),
        duration_seconds = np.random.uniform(30, 600, n),
        connection_count = np.random.randint(1, 3, n),
        failed_connection_count = np.random.randint(0, 2, n),
        syn_count      = np.random.randint(1, 3, n),
        rst_count      = np.zeros(n, dtype=int),
        label          = ["NORMAL"] * n,
        scenario_type  = ["NORMAL_SSH"] * n,
    )


def make_normal_email(n):
    return dict(
        protocol       = np.random.choice(["TCP"], n),
        source_port    = np.random.randint(49152, 65535, n),
        destination_port = np.random.choice([25, 110, 143, 587], n),
        packet_count   = np.random.randint(10, 80, n),
        byte_count     = np.random.randint(2000, 50000, n),
        duration_seconds = np.random.uniform(1, 30, n),
        connection_count = np.random.randint(1, 4, n),
        failed_connection_count = np.zeros(n, dtype=int),
        syn_count      = np.random.randint(1, 3, n),
        rst_count      = np.zeros(n, dtype=int),
        label          = ["NORMAL"] * n,
        scenario_type  = ["NORMAL_EMAIL"] * n,
    )


def make_normal_db(n):
    return dict(
        protocol       = ["TCP"] * n,
        source_port    = np.random.randint(49152, 65535, n),
        destination_port = np.random.choice([3306, 5432, 1433, 27017], n),
        packet_count   = np.random.randint(10, 100, n),
        byte_count     = np.random.randint(5000, 500000, n),
        duration_seconds = np.random.uniform(0.1, 5, n),
        connection_count = np.random.randint(1, 10, n),
        failed_connection_count = np.zeros(n, dtype=int),
        syn_count      = np.random.randint(1, 5, n),
        rst_count      = np.zeros(n, dtype=int),
        label          = ["NORMAL"] * n,
        scenario_type  = ["NORMAL_DATABASE"] * n,
    )


def make_high_conn_rate(n):
    return dict(
        protocol       = ["TCP"] * n,
        source_port    = np.random.randint(49152, 65535, n),
        destination_port = np.random.choice([80, 443, 22], n),
        packet_count   = np.random.randint(200, 2000, n),
        byte_count     = np.random.randint(50000, 500000, n),
        duration_seconds = np.random.uniform(0.5, 5, n),
        connection_count = np.random.randint(150, 1000, n),   # very high
        failed_connection_count = np.random.randint(10, 100, n),
        syn_count      = np.random.randint(100, 800, n),
        rst_count      = np.random.randint(10, 200, n),
        label          = ["SUSPICIOUS"] * n,
        scenario_type  = ["HIGH_CONNECTION_RATE"] * n,
    )


def make_repeated_fails(n):
    return dict(
        protocol       = ["TCP"] * n,
        source_port    = np.random.randint(49152, 65535, n),
        destination_port = np.random.choice([22, 3389, 21, 23], n),
        packet_count   = np.random.randint(50, 500, n),
        byte_count     = np.random.randint(5000, 50000, n),
        duration_seconds = np.random.uniform(1, 30, n),
        connection_count = np.random.randint(20, 200, n),
        failed_connection_count = np.random.randint(30, 150, n),  # many failures
        syn_count      = np.random.randint(30, 200, n),
        rst_count      = np.random.randint(20, 150, n),
        label          = ["SUSPICIOUS"] * n,
        scenario_type  = ["REPEATED_FAILED_CONNECTIONS"] * n,
    )


def make_multi_port(n):
    return dict(
        protocol       = ["TCP"] * n,
        source_port    = np.random.randint(49152, 65535, n),
        destination_port = np.random.randint(1, 65535, n),   # wide port range
        packet_count   = np.random.randint(1, 5, n),
        byte_count     = np.random.randint(40, 500, n),
        duration_seconds = np.random.uniform(0.001, 0.5, n),
        connection_count = np.random.randint(50, 500, n),
        failed_connection_count = np.random.randint(40, 450, n),
        syn_count      = np.random.randint(50, 500, n),
        rst_count      = np.random.randint(40, 490, n),
        label          = ["SUSPICIOUS"] * n,
        scenario_type  = ["MULTI_PORT_PROBING_PATTERN"] * n,
    )


def make_syn_heavy(n):
    return dict(
        protocol       = ["TCP"] * n,
        source_port    = np.random.randint(49152, 65535, n),
        destination_port = np.random.choice([80, 443], n),
        packet_count   = np.random.randint(500, 5000, n),
        byte_count     = np.random.randint(20000, 200000, n),
        duration_seconds = np.random.uniform(0.1, 2, n),
        connection_count = np.random.randint(200, 2000, n),
        failed_connection_count = np.random.randint(100, 1000, n),
        syn_count      = np.random.randint(400, 4000, n),     # very high SYN
        rst_count      = np.random.randint(50, 500, n),
        label          = ["SUSPICIOUS"] * n,
        scenario_type  = ["SYN_HEAVY_PATTERN"] * n,
    )


def make_unusual_port(n):
    return dict(
        protocol       = np.random.choice(["TCP","UDP"], n),
        source_port    = np.random.randint(49152, 65535, n),
        destination_port = np.random.choice(UNUSUAL_PORTS, n),
        packet_count   = np.random.randint(5, 100, n),
        byte_count     = np.random.randint(500, 50000, n),
        duration_seconds = np.random.uniform(0.1, 10, n),
        connection_count = np.random.randint(1, 20, n),
        failed_connection_count = np.random.randint(0, 10, n),
        syn_count      = np.random.randint(1, 20, n),
        rst_count      = np.random.randint(0, 10, n),
        label          = ["SUSPICIOUS"] * n,
        scenario_type  = ["UNUSUAL_PORT_ACTIVITY"] * n,
    )


def make_high_volume(n):
    return dict(
        protocol       = ["TCP"] * n,
        source_port    = np.random.randint(49152, 65535, n),
        destination_port = np.random.choice([80, 443], n),
        packet_count   = np.random.randint(10000, 100000, n),
        byte_count     = np.random.randint(10_000_000, 100_000_000, n),  # very large
        duration_seconds = np.random.uniform(1, 60, n),
        connection_count = np.random.randint(50, 500, n),
        failed_connection_count = np.random.randint(0, 50, n),
        syn_count      = np.random.randint(50, 500, n),
        rst_count      = np.random.randint(5, 100, n),
        label          = ["SUSPICIOUS"] * n,
        scenario_type  = ["HIGH_TRAFFIC_VOLUME"] * n,
    )


# ── Generators map ─────────────────────────────────────────────────────────────
NORMAL_GENERATORS = [
    (make_normal_web,   int(n_normal * 0.35)),
    (make_normal_dns,   int(n_normal * 0.22)),
    (make_normal_ssh,   int(n_normal * 0.15)),
    (make_normal_email, int(n_normal * 0.15)),
    (make_normal_db,    int(n_normal * 0.13)),
]

SUSPICIOUS_GENERATORS = [
    (make_high_conn_rate,  int(n_suspicious * 0.28)),
    (make_repeated_fails,  int(n_suspicious * 0.23)),
    (make_multi_port,      int(n_suspicious * 0.20)),
    (make_syn_heavy,       int(n_suspicious * 0.15)),
    (make_unusual_port,    int(n_suspicious * 0.09)),
    (make_high_volume,     int(n_suspicious * 0.05)),
]


def generate_flows():
    records = []
    base_ts = datetime(2024, 1, 1, 0, 0, 0)

    for gen_fn, count in NORMAL_GENERATORS + SUSPICIOUS_GENERATORS:
        if count <= 0:
            continue
        d = gen_fn(count)
        n = count
        flow_records = []
        for i in range(n):
            ts      = base_ts + timedelta(seconds=np.random.randint(0, 90*86400))
            src_ip  = np.random.choice(ALL_SRC_IPS)
            dst_ip  = np.random.choice(ALL_DST_IPS)
            pkt_cnt = int(d["packet_count"][i])
            byte_c  = int(d["byte_count"][i])
            dur     = float(d["duration_seconds"][i])
            avg_pkt = round(byte_c / max(pkt_cnt, 1), 2)

            flow_records.append({
                "flow_id":                  f"FLW{len(records)+i:07d}",
                "timestamp":               ts.strftime("%Y-%m-%d %H:%M:%S"),
                "source_ip":               src_ip,
                "destination_ip":          dst_ip,
                "source_port":             int(d["source_port"][i]),
                "destination_port":        int(d["destination_port"][i]),
                "protocol":                d["protocol"][i],
                "packet_count":            pkt_cnt,
                "byte_count":              byte_c,
                "duration_seconds":        round(dur, 4),
                "connection_count":        int(d["connection_count"][i]),
                "failed_connection_count": int(d["failed_connection_count"][i]),
                "syn_count":               int(d["syn_count"][i]),
                "rst_count":               int(d["rst_count"][i]),
                "average_packet_size":     avg_pkt,
                "label":                   d["label"][i],
                "scenario_type":           d["scenario_type"][i],
            })
        records.extend(flow_records)

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔧 Generating synthetic network traffic dataset …")
    print("   ⚠️  All IPs use RFC 5737 documentation ranges (non-routable)")
    print("   ⚠️  No real network packets are generated\n")

    df = generate_flows()

    csv_path = OUTPUT_DIR / "network_traffic.csv"
    pq_path  = OUTPUT_DIR / "network_traffic.parquet"

    df.to_csv(csv_path, index=False)
    print(f"✅ CSV saved     → {csv_path}  ({len(df):,} rows)")

    df.to_parquet(pq_path, index=False)
    print(f"✅ Parquet saved → {pq_path}")

    n_susp = (df["label"] == "SUSPICIOUS").sum()
    n_norm = (df["label"] == "NORMAL").sum()

    print(f"\n📊 Dataset Summary")
    print(f"   Total flows    : {len(df):,}")
    print(f"   Normal         : {n_norm:,}  ({n_norm/len(df)*100:.1f}%)")
    print(f"   Suspicious     : {n_susp:,}  ({n_susp/len(df)*100:.1f}%)")
    print(f"\n   Scenario breakdown:")
    print(df["scenario_type"].value_counts().to_string())
    print(f"\n   Protocol breakdown:")
    print(df["protocol"].value_counts().to_string())
    print(f"\n   Columns: {list(df.columns)}")