"""
ml/train_classifier.py
───────────────────────
Phase 2 → XGBoost ML Classifier

Trains a supervised XGBoost model on labelled flows.
Also builds the hybrid risk scorer combining:
  rule_score × 0.40 + anomaly_score × 0.30 + ml_score × 0.30

Run:
    python ml/train_classifier.py
"""

import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score,
    roc_curve, average_precision_score, precision_recall_curve,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_DIR   = Path("data")
MODEL_DIR  = Path("models")
OUTPUT_DIR = Path("outputs")

# ── Load ──────────────────────────────────────────────────────────────────────
print("📂 Loading data …")
df_raw       = pd.read_parquet(DATA_DIR / "network_traffic.parquet")
features     = pd.read_parquet(DATA_DIR / "features.parquet")
rule_scores  = pd.read_parquet(DATA_DIR / "rule_scores.parquet")
anomaly      = pd.read_parquet(DATA_DIR / "anomaly_scores.parquet")
feature_cols = joblib.load(MODEL_DIR / "feature_list.joblib")

X = features[feature_cols].fillna(0)
y = (df_raw["label"] == "SUSPICIOUS").astype(int)

print(f"   {len(X):,} flows  |  Suspicious: {y.sum():,}  ({y.mean()*100:.1f}%)")

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

scaler  = StandardScaler()
Xtr_sc  = scaler.fit_transform(X_train)
Xte_sc  = scaler.transform(X_test)

# ── Train XGBoost ─────────────────────────────────────────────────────────────
print("\n🏋️  Training XGBoost classifier …")
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)
model.fit(Xtr_sc, y_train, verbose=False)

y_prob  = model.predict_proba(Xte_sc)[:, 1]
threshs = np.linspace(0.01, 0.99, 200)
f1s     = [f1_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in threshs]
best_t  = threshs[np.argmax(f1s)]
y_pred  = (y_prob >= best_t).astype(int)

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc  = average_precision_score(y_test, y_prob)
f1      = f1_score(y_test, y_pred, zero_division=0)
prec    = precision_score(y_test, y_pred, zero_division=0)
rec     = recall_score(y_test, y_pred, zero_division=0)

print(f"\n{'='*55}")
print("📊 ML Classifier Results")
print(f"{'='*55}")
print(f"   ROC-AUC   : {roc_auc:.4f}")
print(f"   PR-AUC    : {pr_auc:.4f}")
print(f"   F1 Score  : {f1:.4f}")
print(f"   Precision : {prec:.4f}")
print(f"   Recall    : {rec:.4f}")
print(f"   Threshold : {best_t:.3f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['NORMAL','SUSPICIOUS'])}")

# ── Score ALL flows for hybrid engine ─────────────────────────────────────────
print("🔧 Scoring all flows for hybrid engine …")
X_all_sc = scaler.transform(X)
ml_probs  = model.predict_proba(X_all_sc)[:, 1] * 100  # 0–100

RULE_W    = 0.40
ANOMALY_W = 0.30
ML_W      = 0.30

hybrid_score = (
    RULE_W    * rule_scores["rule_score"].values +
    ANOMALY_W * anomaly["anomaly_score"].values +
    ML_W      * ml_probs
).clip(0, 100).round(2)

def to_severity(score):
    if score >= 80:  return "CRITICAL"
    if score >= 65:  return "HIGH"
    if score >= 40:  return "MEDIUM"
    if score >= 20:  return "LOW"
    return "INFO"

severity = [to_severity(s) for s in hybrid_score]

final = pd.DataFrame({
    "flow_id":       df_raw["flow_id"],
    "timestamp":     df_raw["timestamp"],
    "source_ip":     df_raw["source_ip"],
    "destination_ip":df_raw["destination_ip"],
    "source_port":   df_raw["source_port"],
    "destination_port": df_raw["destination_port"],
    "protocol":      df_raw["protocol"],
    "label":         df_raw["label"],
    "scenario_type": df_raw["scenario_type"],
    "rule_score":    rule_scores["rule_score"].values,
    "anomaly_score": anomaly["anomaly_score"].values,
    "ml_score":      ml_probs.round(2),
    "hybrid_score":  hybrid_score,
    "severity":      severity,
    "is_alert":      hybrid_score >= 40,
})

final.to_parquet(DATA_DIR / "final_scores.parquet", index=False)

# ── Save models ───────────────────────────────────────────────────────────────
joblib.dump(model,   MODEL_DIR / "xgb_ids_model.joblib")
joblib.dump(scaler,  MODEL_DIR / "ml_scaler.joblib")
joblib.dump(best_t,  MODEL_DIR / "ml_threshold.joblib")
joblib.dump({
    "roc_auc": roc_auc, "pr_auc": pr_auc,
    "f1": f1, "precision": prec, "recall": rec,
    "threshold": best_t,
}, MODEL_DIR / "eval_metrics.joblib")

n_alerts = final["is_alert"].sum()
print(f"\n   Total alerts  : {n_alerts:,}  ({n_alerts/len(final)*100:.1f}%)")
print(f"   Critical      : {(final['severity']=='CRITICAL').sum():,}")
print(f"   High          : {(final['severity']=='HIGH').sum():,}")
print(f"   Medium        : {(final['severity']=='MEDIUM').sum():,}")

# ── Charts ────────────────────────────────────────────────────────────────────
print("\n📊 Generating evaluation charts …")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("IDS ML Classifier — XGBoost", fontsize=13, fontweight="bold")

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr, tpr, color="#e74c3c", lw=2, label=f"ROC-AUC={roc_auc:.4f}")
axes[0].plot([0,1],[0,1],"k--",lw=1)
axes[0].fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")
axes[0].set_title("ROC Curve"); axes[0].legend()

pc, rc, _ = precision_recall_curve(y_test, y_prob)
axes[1].plot(rc, pc, color="#3498db", lw=2, label=f"PR-AUC={pr_auc:.4f}")
axes[1].fill_between(rc, pc, alpha=0.1, color="#3498db")
axes[1].set_title("Precision-Recall Curve"); axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_roc_pr_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# Confusion matrix
fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["NORMAL","SUSPICIOUS"],
            yticklabels=["NORMAL","SUSPICIOUS"], ax=ax)
ax.set_title(f"Confusion Matrix  (threshold={best_t:.3f})", fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# Hybrid score distribution
fig, ax = plt.subplots(figsize=(12, 5))
colors = {"NORMAL":"#2ecc71","SUSPICIOUS":"#e74c3c"}
for label, grp in final.groupby("label"):
    ax.hist(grp["hybrid_score"], bins=60, alpha=0.6, label=label,
            color=colors[label], density=True)
for thresh, label, color in [(40,"MEDIUM","#f0a500"),(65,"HIGH","#e74c3c"),(80,"CRITICAL","#c0392b")]:
    ax.axvline(thresh, linestyle="--", color=color, lw=1.5, label=f"{label}={thresh}")
ax.set_title("Hybrid Risk Score Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Hybrid Score (0–100)"); ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_hybrid_score_dist.png", dpi=150, bbox_inches="tight")
plt.close()

# Feature importance
fig, ax = plt.subplots(figsize=(10, 8))
imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values().tail(20)
imp.plot(kind="barh", ax=ax, color="#3498db", edgecolor="white")
ax.set_title("XGBoost Feature Importance (Top 20)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

print("   ✅ outputs/02_roc_pr_curves.png")
print("   ✅ outputs/03_confusion_matrix.png")
print("   ✅ outputs/04_hybrid_score_dist.png")
print("   ✅ outputs/05_feature_importance.png")

print(f"\n{'='*55}")
print(f"✅ Phase 2 complete!")
print(f"   ROC-AUC={roc_auc:.4f} | F1={f1:.4f}")
print(f"   Next → python backend/api.py  (Phase 3)")
print(f"{'='*55}")