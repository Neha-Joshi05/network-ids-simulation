# 🛡️ Network Intrusion Detection System (IDS) Simulation

A production-grade, end-to-end cybersecurity simulation system for detecting network intrusions using a **Hybrid Detection Engine** — combining Signature-Based Rules, Anomaly Detection (Isolation Forest), and Machine Learning (XGBoost) — served via FastAPI with a live SOC dashboard.

> ⚠️ **Educational Project** — All IPs use RFC 5737 documentation ranges. No real packets are generated or captured.

---

## 🖥️ Live Demo

🔗 **SOC Dashboard** → [your-project.vercel.app](https://your-project.vercel.app)

---

## 📸 Overview

```
Raw Network Flow → Feature Extraction → Rule Engine (40%)
                                      + Anomaly Detector (30%)  → Hybrid Score → Alert
                                      + ML Classifier (30%)
```

**Hybrid Risk Score Formula:**
```
Hybrid Score = Rule Score × 0.40 + Anomaly Score × 0.30 + ML Score × 0.30
```

| Severity | Score Range | Action |
|---|---|---|
| 🟢 INFO | 0–19 | Monitor |
| 🔵 LOW | 20–39 | Log |
| 🟡 MEDIUM | 40–64 | Investigate |
| 🟠 HIGH | 65–79 | Escalate |
| 🔴 CRITICAL | 80–100 | Immediate Response |

---

## 🗂️ Project Structure

```
Network-IDS-Simulation/
├── main.py                        # Pipeline runner
├── verify_setup.py                # Environment checker
├── requirements.txt
├── .gitignore
├── .env.example
│
├── simulator/
│   ├── generate_dataset.py        # 6,000 synthetic network flows
│   └── 01_ingest.py              # Schema enforcement + validation
│
├── ids/
│   ├── feature_extractor.py      # 30 engineered features
│   ├── rule_engine.py            # 12 signature-based detection rules
│   └── anomaly_detector.py       # Isolation Forest (unsupervised)
│
├── ml/
│   └── train_classifier.py       # XGBoost + hybrid score engine
│
├── backend/
│   └── api.py                    # FastAPI (6 endpoints)
│
├── frontend/
│   └── dashboard.html            # SENTINEL SOC Dashboard
│
├── data/                         # Generated at runtime (gitignored)
├── models/                       # Saved models (gitignored)
└── outputs/                      # Charts (gitignored)
```

---

## ⚙️ Setup

### 1. Clone & create virtual environment

```bash
git clone https://github.com/your-username/Network-IDS-Simulation.git
cd Network-IDS-Simulation

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install pyarrow
```

### 3. Verify environment

```bash
python verify_setup.py
```

---

## 🚀 Running the Pipeline

```bash
# Phase 1 — Generate & ingest data
python simulator/generate_dataset.py
python simulator/01_ingest.py

# Phase 2 — IDS Engine
python ids/feature_extractor.py
python ids/rule_engine.py
python ids/anomaly_detector.py
python ml/train_classifier.py

# Phase 3 — Serve API
uvicorn backend.api:app --reload --port 8000

# Open dashboard
# Double-click frontend/dashboard.html
```

---

## 📊 Dataset

Synthetic dataset of **6,000 network flow records** across 11 traffic scenarios:

| Scenario | Type | Description |
|---|---|---|
| NORMAL_WEB | Normal | HTTP/HTTPS browsing |
| NORMAL_DNS | Normal | DNS resolution |
| NORMAL_SSH | Normal | Legitimate SSH sessions |
| NORMAL_EMAIL | Normal | SMTP/IMAP traffic |
| NORMAL_DATABASE | Normal | DB queries |
| HIGH_CONNECTION_RATE | Suspicious | DoS indicator |
| REPEATED_FAILED_CONNECTIONS | Suspicious | Brute-force / credential stuffing |
| MULTI_PORT_PROBING_PATTERN | Suspicious | Port scanning |
| SYN_HEAVY_PATTERN | Suspicious | SYN flood attack |
| UNUSUAL_PORT_ACTIVITY | Suspicious | C2 / malware communication |
| HIGH_TRAFFIC_VOLUME | Suspicious | Data exfiltration |

**Class distribution:** ~65% Normal · ~35% Suspicious

---

## 🤖 Detection Engine

### Layer 1 — Rule Engine (12 Signatures)

| Rule | Trigger | Score |
|---|---|---|
| SYN_FLOOD_PATTERN | SYN ratio > 0.8, count > 200 | 95 |
| PORT_SCAN_PATTERN | High conn + fail rate, tiny packets | 88 |
| HIGH_CONNECTION_RATE | Connection count > 500 | 90 |
| SSH_BRUTEFORCE | Port 22 + failed conns > 15 | 80 |
| UNUSUAL_DESTINATION_PORT | Known malware ports | 70 |
| HIGH_TRAFFIC_VOLUME | Bytes > 50MB | 80 |
| REPEATED_FAILED_CONNECTIONS | Fail ratio > 0.6 | 85 |
| RST_FLOOD | RST ratio > 0.5 | 60 |
| RAPID_CONNECTION_RATE | Conns/sec > 100 | 85 |
| + 3 more | … | … |

### Layer 2 — Anomaly Detector

- **Algorithm:** Isolation Forest (200 estimators)
- **Training:** NORMAL flows only (unsupervised)
- **Output:** Anomaly score 0–100

### Layer 3 — ML Classifier

- **Algorithm:** XGBoost
- **Features:** 30 engineered features
- **Output:** Pass probability → ML score 0–100

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | 0.97+ |
| PR-AUC | 0.96+ |
| F1 Score | 0.94+ |
| Precision | 0.95+ |
| Recall | 0.93+ |

---

## 🌐 API

Start:
```bash
uvicorn backend.api:app --reload --port 8000
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/metrics` | Model metrics + alert stats |
| `GET` | `/alerts` | Paginated alert feed |
| `GET` | `/alerts/{id}` | Single alert detail |
| `POST` | `/analyze` | Score a single flow |
| `POST` | `/analyze/batch` | Score up to 1,000 flows |

### Docs
- Swagger UI → [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc → [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Example Request

```bash
Invoke-RestMethod -Method Post -Uri http://localhost:8000/analyze `
  -ContentType "application/json" `
  -Body '{
    "source_ip": "192.0.2.10",
    "destination_ip": "203.0.113.5",
    "source_port": 52341,
    "destination_port": 4444,
    "protocol": "TCP",
    "packet_count": 850,
    "byte_count": 42000,
    "duration_seconds": 1.2,
    "connection_count": 320,
    "failed_connection_count": 180,
    "syn_count": 290,
    "rst_count": 140
  }'
```

### Example Response

```json
{
  "flow_id": "FLW-142531089",
  "source_ip": "192.0.2.10",
  "destination_ip": "203.0.113.5",
  "protocol": "TCP",
  "rule_score": 87.5,
  "anomaly_score": 76.2,
  "ml_score": 91.4,
  "hybrid_score": 84.8,
  "severity": "CRITICAL",
  "is_alert": true,
  "triggered_rules": [
    "HIGH_CONNECTION_RATE",
    "REPEATED_FAILED_CONNECTIONS",
    "UNUSUAL_DESTINATION_PORT"
  ],
  "recommendations": [
    "🚨 Enable SYN cookies and rate-limit SYN packets at firewall",
    "🔒 Credential stuffing suspected — enforce MFA immediately",
    "⚠️ Inspect traffic — connection to known malware/C2 port"
  ],
  "timestamp": "2024-03-15T14:22:11.042Z"
}
```

---

## 🖥️ SENTINEL SOC Dashboard

Open `frontend/dashboard.html` while the API is running.

**Features:**
- 🎨 Dark cyber aesthetic — Orbitron font, scanlines, neon glow effects
- 🔴 Live threat level banner — NOMINAL → CRITICAL (blinks on active threat)
- ⚡ 3 quick presets — SYN Flood · Port Scan · Normal Traffic
- 📊 Score decomposition — Rule / Anomaly / ML scores shown separately
- 🌡️ Animated hybrid risk meter — shifts green → amber → red
- 🏷️ Triggered signature tags — all matched rules shown
- 💡 Response recommendations — specific analyst actions
- 📡 Live alert feed — color-coded severity with animated rows
- 📈 Score history chart — last 20 flows as neon bars
- 🔢 Session counters — Critical / High / Medium tallied live

---

## 📈 EDA & Model Charts

Generated to `outputs/` during the pipeline run.

### Anomaly Detection
![Anomaly Detection](outputs/01_anomaly_detection.png)

### ROC & PR Curves
![ROC PR Curves](outputs/02_roc_pr_curves.png)

### Confusion Matrix
![Confusion Matrix](outputs/03_confusion_matrix.png)

### Hybrid Score Distribution
![Hybrid Score](outputs/04_hybrid_score_dist.png)

### Feature Importance
![Feature Importance](outputs/05_feature_importance.png)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data Simulation | Python, NumPy, pandas |
| Feature Engineering | Custom pipeline (30 features) |
| Rule Engine | Hand-crafted signatures (12 rules) |
| Anomaly Detection | scikit-learn Isolation Forest |
| ML Classifier | XGBoost |
| API | FastAPI, uvicorn, pydantic |
| Persistence | joblib, pyarrow |
| Dashboard | HTML/CSS/JS (Orbitron + Rajdhani fonts) |

---

## 🏢 Real-World Applications

Systems like this power:

- **Palo Alto Networks** — ML-based threat detection in next-gen firewalls
- **Cisco Secure** — network anomaly detection at enterprise scale
- **Darktrace** — unsupervised AI detecting novel attack patterns
- **CrowdStrike** — behavioural IOC matching in real time
- **CERT-In (India)** — national-level network monitoring infrastructure
- **IBM QRadar** — SIEM with hybrid rule + ML detection engine
- **Splunk Enterprise Security** — correlation rules + anomaly scoring

---
## 🙏 Acknowledgements

- [XGBoost](https://xgboost.readthedocs.io/)
- [scikit-learn](https://scikit-learn.org/) — Isolation Forest
- [FastAPI](https://fastapi.tiangolo.com/)
- [Orbitron Font](https://fonts.google.com/specimen/Orbitron)
- MITRE ATT&CK Framework — attack pattern reference

---
# Author
**Neha Joshi**
[GitHub]: 
[LinkedIn]: