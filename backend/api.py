"""
backend/api.py
──────────────
Phase 3 → FastAPI IDS REST API

Endpoints:
    GET  /              → health check
    GET  /metrics       → model performance + alert stats
    GET  /alerts        → paginated alert feed
    GET  /alerts/{id}   → single alert detail
    POST /analyze       → score a single flow
    POST /analyze/batch → score up to 1000 flows

Run:
    uvicorn backend.api:app --reload --port 8000
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Load artifacts ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODEL_DIR  = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"

ml_model      = joblib.load(MODEL_DIR / "xgb_ids_model.joblib")
ml_scaler     = joblib.load(MODEL_DIR / "ml_scaler.joblib")
ml_threshold  = joblib.load(MODEL_DIR / "ml_threshold.joblib")
iso_model     = joblib.load(MODEL_DIR / "isolation_forest.joblib")
iso_scaler    = joblib.load(MODEL_DIR / "anomaly_scaler.joblib")
feature_cols  = joblib.load(MODEL_DIR / "feature_list.joblib")
anomaly_feats = joblib.load(MODEL_DIR / "anomaly_features.joblib")
eval_metrics  = joblib.load(MODEL_DIR / "eval_metrics.joblib")

# Pre-load final scored dataset for alert feed
final_df = pd.read_parquet(DATA_DIR / "final_scores.parquet")
final_df["timestamp"] = pd.to_datetime(final_df["timestamp"])
ALERTS_DF = final_df[final_df["is_alert"]].sort_values("hybrid_score", ascending=False).reset_index(drop=True)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Network IDS — Simulation API",
    description = "Hybrid IDS: Rule-based + Anomaly + ML scoring engine",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ── Constants ──────────────────────────────────────────────────────────────────
UNUSUAL_PORTS = {4444, 6667, 31337, 1337, 12345, 54321, 9999, 7777}
COMMON_PORTS  = {80, 443, 53, 22, 25, 110, 143, 3306, 5432, 8080, 8443}
RULE_W, ANOMALY_W, ML_W = 0.40, 0.30, 0.30


# ── Schemas ────────────────────────────────────────────────────────────────────
class FlowInput(BaseModel):
    source_ip:               str   = Field(...,  description="Source IP (use RFC 5737 ranges)")
    destination_ip:          str   = Field(...,  description="Destination IP")
    source_port:             int   = Field(...,  ge=0, le=65535)
    destination_port:        int   = Field(...,  ge=0, le=65535)
    protocol:                str   = Field(...,  description="TCP/UDP/ICMP/DNS/HTTP/HTTPS/SSH/FTP")
    packet_count:            int   = Field(...,  ge=0)
    byte_count:              int   = Field(...,  ge=0)
    duration_seconds:        float = Field(...,  ge=0)
    connection_count:        int   = Field(1,    ge=0)
    failed_connection_count: int   = Field(0,    ge=0)
    syn_count:               int   = Field(0,    ge=0)
    rst_count:               int   = Field(0,    ge=0)

    class Config:
        json_schema_extra = {"example": {
            "source_ip": "192.0.2.10", "destination_ip": "203.0.113.5",
            "source_port": 52341, "destination_port": 4444,
            "protocol": "TCP", "packet_count": 850,
            "byte_count": 42000, "duration_seconds": 1.2,
            "connection_count": 320, "failed_connection_count": 180,
            "syn_count": 290, "rst_count": 140,
        }}


class AnalysisResponse(BaseModel):
    flow_id:        str
    source_ip:      str
    destination_ip: str
    protocol:       str
    rule_score:     float
    anomaly_score:  float
    ml_score:       float
    hybrid_score:   float
    severity:       str
    is_alert:       bool
    triggered_rules: list[str]
    recommendations: list[str]
    timestamp:      str


class BatchRequest(BaseModel):
    flows: list[FlowInput] = Field(..., max_items=1000)


# ── Feature engineering ────────────────────────────────────────────────────────
def build_features(f: FlowInput) -> dict:
    avg_pkt   = f.byte_count / max(f.packet_count, 1)
    fail_rat  = f.failed_connection_count / (f.connection_count + 1)
    syn_rat   = f.syn_count / (f.packet_count + 1)
    rst_rat   = f.rst_count / (f.packet_count + 1)
    bps       = f.byte_count / (f.duration_seconds + 0.001)
    pps       = f.packet_count / (f.duration_seconds + 0.001)
    bpc       = f.byte_count / (f.connection_count + 1)

    return {
        "packet_count":            f.packet_count,
        "byte_count":              f.byte_count,
        "duration_seconds":        f.duration_seconds,
        "connection_count":        f.connection_count,
        "failed_connection_count": f.failed_connection_count,
        "syn_count":               f.syn_count,
        "rst_count":               f.rst_count,
        "average_packet_size":     avg_pkt,
        "source_port":             f.source_port,
        "destination_port":        f.destination_port,
        "failed_conn_ratio":       min(fail_rat, 1),
        "syn_ratio":               min(syn_rat, 1),
        "rst_ratio":               min(rst_rat, 1),
        "bytes_per_second":        max(bps, 0),
        "packets_per_second":      max(pps, 0),
        "bytes_per_connection":    bpc,
        "log_byte_count":          np.log1p(f.byte_count),
        "log_packet_count":        np.log1p(f.packet_count),
        "log_connection_count":    np.log1p(f.connection_count),
        "is_unusual_port":         int(f.destination_port in UNUSUAL_PORTS),
        "is_common_port":          int(f.destination_port in COMMON_PORTS),
        "is_ephemeral_src_port":   int(f.source_port > 49151),
        "port_diff":               abs(f.destination_port - f.source_port),
        "is_tcp":                  int(f.protocol == "TCP"),
        "is_udp":                  int(f.protocol == "UDP"),
        "is_icmp":                 int(f.protocol == "ICMP"),
        "is_dns":                  int(f.protocol == "DNS"),
        "is_ssh":                  int(f.protocol == "SSH"),
        "high_conn_rate":          int(f.connection_count > 100),
        "high_fail_rate":          int(fail_rat > 0.4),
        "high_syn_rate":           int(syn_rat > 0.6),
        "burst_traffic":           int(bps > 1_000_000),
        "risk_flag_sum":           sum([
            int(f.connection_count > 100),
            int(fail_rat > 0.4),
            int(syn_rat > 0.6),
            int(bps > 1_000_000),
            int(f.destination_port in UNUSUAL_PORTS),
        ]),
    }


def compute_rule_score(f: FlowInput) -> tuple[float, list[str]]:
    scores   = []
    triggers = []
    fail_rat = f.failed_connection_count / (f.connection_count + 1)
    syn_rat  = f.syn_count / (f.packet_count + 1)
    rst_rat  = f.rst_count / (f.packet_count + 1)
    bps      = f.byte_count / (f.duration_seconds + 0.001)
    cps      = f.connection_count / (f.duration_seconds + 0.001)

    if f.connection_count > 500:   scores.append(90); triggers.append("HIGH_CONNECTION_RATE")
    elif f.connection_count > 200: scores.append(65); triggers.append("ELEVATED_CONNECTION_RATE")
    elif f.connection_count > 100: scores.append(40); triggers.append("MODERATE_CONNECTION_RATE")

    if fail_rat > 0.6 and f.failed_connection_count > 20:
        scores.append(85); triggers.append("REPEATED_FAILED_CONNECTIONS")
    elif fail_rat > 0.4 and f.failed_connection_count > 10:
        scores.append(55); triggers.append("ELEVATED_FAILED_CONNECTIONS")

    if syn_rat > 0.8 and f.syn_count > 200:
        scores.append(95); triggers.append("SYN_FLOOD_PATTERN")
    elif syn_rat > 0.6 and f.syn_count > 50:
        scores.append(70); triggers.append("HIGH_SYN_RATE")

    if f.connection_count > 50 and f.failed_connection_count > 40 and f.packet_count < 10:
        scores.append(88); triggers.append("PORT_SCAN_PATTERN")

    if f.destination_port in UNUSUAL_PORTS:
        scores.append(70); triggers.append("UNUSUAL_DESTINATION_PORT")

    if f.byte_count > 50_000_000:   scores.append(80); triggers.append("HIGH_TRAFFIC_VOLUME")
    elif f.byte_count > 10_000_000: scores.append(50); triggers.append("ELEVATED_TRAFFIC_VOLUME")

    if rst_rat > 0.5:  scores.append(60); triggers.append("HIGH_RST_RATE")

    if f.destination_port == 22 and f.failed_connection_count > 15:
        scores.append(80); triggers.append("SSH_BRUTEFORCE_PATTERN")

    if cps > 100: scores.append(85); triggers.append("RAPID_CONNECTION_RATE")

    if not scores: return 0.0, []

    top2  = sorted(scores)[-2:]
    rest  = sorted(scores)[:-2]
    score = (0.70 * np.mean(top2) + (0.30 * np.mean(rest) if rest else 0))
    return round(min(score, 100), 2), triggers


def compute_anomaly_score(feat_dict: dict) -> float:
    available = {k: feat_dict[k] for k in anomaly_feats if k in feat_dict}
    row = pd.DataFrame([available])[anomaly_feats]
    scaled = iso_scaler.transform(row)
    raw = iso_model.decision_function(scaled)[0]
    # Normalise to 0–100 using training distribution heuristic
    score = max(0, min(100, (0.5 - raw) * 100))
    return round(score, 2)


def get_severity(score: float) -> str:
    if score >= 80: return "CRITICAL"
    if score >= 65: return "HIGH"
    if score >= 40: return "MEDIUM"
    if score >= 20: return "LOW"
    return "INFO"


def get_recommendations(triggers: list[str], severity: str) -> list[str]:
    recs = []
    if "SYN_FLOOD_PATTERN" in triggers:
        recs.append("🚨 Enable SYN cookies and rate-limit SYN packets at firewall")
    if "PORT_SCAN_PATTERN" in triggers:
        recs.append("🔍 Block source IP — active port scanning detected")
    if "SSH_BRUTEFORCE_PATTERN" in triggers:
        recs.append("🔐 Temporarily block source IP — SSH brute-force in progress")
    if "UNUSUAL_DESTINATION_PORT" in triggers:
        recs.append("⚠️ Inspect traffic — connection to known malware/C2 port")
    if "HIGH_TRAFFIC_VOLUME" in triggers:
        recs.append("📊 Check for data exfiltration — abnormal outbound volume")
    if "REPEATED_FAILED_CONNECTIONS" in triggers:
        recs.append("🔒 Credential stuffing suspected — enforce MFA immediately")
    if severity in ("CRITICAL","HIGH") and not recs:
        recs.append("🚨 Immediate investigation required — high hybrid risk score")
    if not recs:
        recs.append("ℹ️ Monitor flow — low-level anomaly detected")
    return recs[:3]


def score_flow(f: FlowInput) -> AnalysisResponse:
    feat_dict   = build_features(f)
    rule_score, triggers = compute_rule_score(f)
    anomaly_score = compute_anomaly_score(feat_dict)

    feat_row  = pd.DataFrame([feat_dict])[feature_cols]
    feat_sc   = ml_scaler.transform(feat_row)
    ml_prob   = float(ml_model.predict_proba(feat_sc)[0, 1]) * 100

    hybrid = round(RULE_W * rule_score + ANOMALY_W * anomaly_score + ML_W * ml_prob, 2)
    sev    = get_severity(hybrid)

    return AnalysisResponse(
        flow_id         = f"FLW-{datetime.utcnow().strftime('%H%M%S%f')[:12]}",
        source_ip       = f.source_ip,
        destination_ip  = f.destination_ip,
        protocol        = f.protocol,
        rule_score      = rule_score,
        anomaly_score   = anomaly_score,
        ml_score        = round(ml_prob, 2),
        hybrid_score    = hybrid,
        severity        = sev,
        is_alert        = hybrid >= 40,
        triggered_rules = triggers,
        recommendations = get_recommendations(triggers, sev),
        timestamp       = datetime.utcnow().isoformat(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    return {
        "status":    "online",
        "service":   "Network IDS Simulation API",
        "version":   "1.0.0",
        "engine":    "Hybrid: Rule(40%) + Anomaly(30%) + ML(30%)",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/metrics", tags=["Model"])
def get_metrics():
    sev_counts = ALERTS_DF["severity"].value_counts().to_dict()
    return {
        "model":   "XGBoost + Isolation Forest + Rule Engine",
        "ml_metrics": {
            "roc_auc":   round(eval_metrics["roc_auc"],   4),
            "pr_auc":    round(eval_metrics["pr_auc"],    4),
            "f1_score":  round(eval_metrics["f1"],        4),
            "precision": round(eval_metrics["precision"], 4),
            "recall":    round(eval_metrics["recall"],    4),
        },
        "dataset": {
            "total_flows":  len(final_df),
            "total_alerts": len(ALERTS_DF),
            "alert_rate":   round(len(ALERTS_DF) / len(final_df) * 100, 2),
        },
        "severity_breakdown": sev_counts,
    }


@app.get("/alerts", tags=["Alerts"])
def get_alerts(
    page:     int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    severity: Optional[str] = Query(None),
    protocol: Optional[str] = Query(None),
):
    df = ALERTS_DF.copy()
    if severity: df = df[df["severity"] == severity.upper()]
    if protocol: df = df[df["protocol"] == protocol.upper()]

    total  = len(df)
    start  = (page - 1) * per_page
    end    = start + per_page
    page_df = df.iloc[start:end]

    return {
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "pages":    (total + per_page - 1) // per_page,
        "alerts": page_df[[
            "flow_id","timestamp","source_ip","destination_ip",
            "protocol","scenario_type","rule_score","anomaly_score",
            "ml_score","hybrid_score","severity","label",
        ]].to_dict(orient="records"),
    }


@app.get("/alerts/{flow_id}", tags=["Alerts"])
def get_alert_detail(flow_id: str):
    row = final_df[final_df["flow_id"] == flow_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Flow not found")
    return row.iloc[0].to_dict()


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
def analyze_flow(flow: FlowInput):
    try:
        return score_flow(flow)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/batch", tags=["Analysis"])
def analyze_batch(req: BatchRequest):
    try:
        results   = [score_flow(f) for f in req.flows]
        n_alerts  = sum(r.is_alert for r in results)
        severities = {}
        for r in results:
            severities[r.severity] = severities.get(r.severity, 0) + 1
        return {
            "total":              len(results),
            "alerts":             n_alerts,
            "alert_rate_pct":     round(n_alerts / len(results) * 100, 2),
            "severity_breakdown": severities,
            "results":            results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)