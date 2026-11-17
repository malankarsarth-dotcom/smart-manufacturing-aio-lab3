"""Use Case 1: Statistical Process Control (SPC) Charting
Compares Pre-AI Optimization (Classical Static Shewhart SPC) vs.
Post-AI Optimization (Robust Baseline + EWMA + Nelson Rules + Isolation Forest).
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from generate_data import generate_spc_dataset


def run_pre_ai_spc(df: pd.DataFrame) -> dict:
    """Classical static SPC using sample mean and sample std (naive 3-sigma)."""
    x = df["diameter_mm"].to_numpy()
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    ucl = mean + 3.0 * std
    lcl = mean - 3.0 * std

    alerts = (x > ucl) | (x < lcl)
    return {
        "mean": mean,
        "std": std,
        "ucl": ucl,
        "lcl": lcl,
        "alerts": alerts,
    }


def compute_ewma(x: np.ndarray, target: float, sigma: float, lam: float = 0.3, L: float = 2.7):
    """Exponentially Weighted Moving Average (EWMA) control chart."""
    n = len(x)
    z = np.zeros(n)
    ucl = np.zeros(n)
    lcl = np.zeros(n)

    prev_z = target
    for i in range(n):
        z[i] = lam * x[i] + (1.0 - lam) * prev_z
        prev_z = z[i]
        factor = np.sqrt((lam / (2.0 - lam)) * (1.0 - (1.0 - lam) ** (2 * (i + 1))))
        ucl[i] = target + L * sigma * factor
        lcl[i] = target - L * sigma * factor

    ewma_alerts = (z > ucl) | (z < lcl)
    return z, ucl, lcl, ewma_alerts


def detect_nelson_trend(x: np.ndarray, centerline: float) -> np.ndarray:
    """Flag 6+ consecutive increasing/decreasing points or 7+ on one side of centerline."""
    n = len(x)
    flags = np.zeros(n, dtype=bool)

    # Rule: 7+ consecutive points on the same side of centerline
    side = np.sign(x - centerline)
    run_len = 0
    prev_side = 0
    for i in range(n):
        if side[i] != 0 and side[i] == prev_side:
            run_len += 1
        else:
            run_len = 1 if side[i] != 0 else 0
            prev_side = side[i]
        if run_len >= 7:
            flags[i] = True

    # Rule: 5+ consecutive strictly increasing or decreasing points
    inc_run = 1
    dec_run = 1
    for i in range(1, n):
        if x[i] > x[i - 1]:
            inc_run += 1
            dec_run = 1
        elif x[i] < x[i - 1]:
            dec_run += 1
            inc_run = 1
        else:
            inc_run = 1
            dec_run = 1
        if inc_run >= 5 or dec_run >= 5:
            flags[i] = True

    return flags


def infer_root_cause(row: pd.Series, robust_mean: float, robust_std: float, ewma_flag: bool, nelson_flag: bool) -> str:
    """Rule-assisted AI root-cause attribution from multivariate telemetry."""
    causes = []
    if row["vibration_rms"] > 0.62 and row["ambient_temp_c"] > 23.5:
        causes.append("Tool wear & thermal/vibration drift")
    elif row["vibration_rms"] > 0.62:
        causes.append("Mechanical vibration / loose fixture")
    elif row["ambient_temp_c"] > 23.5:
        causes.append("Environmental temperature fluctuation")

    if abs(row["diameter_mm"] - robust_mean) > 3.0 * robust_std and not ewma_flag:
        causes.append("Sudden raw material inconsistency or calibration spike")
    elif ewma_flag or nelson_flag:
        causes.append("Sustained process shift / machine calibration drift")

    if not causes:
        causes.append("Multivariate anomaly (operator/process interaction)")
    return "; ".join(dict.fromkeys(causes))


def run_post_ai_spc(df: pd.DataFrame) -> dict:
    """AI-enhanced SPC: Robust baseline + EWMA + Nelson trend + Isolation Forest."""
    x = df["diameter_mm"].to_numpy()

    # Robust center and spread (resistant to outliers inflating sigma)
    robust_mean = float(np.median(x))
    mad = float(np.median(np.abs(x - robust_mean)))
    robust_std = 1.4826 * mad if mad > 0 else float(np.std(x, ddof=1))

    robust_ucl = robust_mean + 3.0 * robust_std
    robust_lcl = robust_mean - 3.0 * robust_std
    limit_alerts = (x > robust_ucl) | (x < robust_lcl)

    # EWMA for early drift detection
    ewma_z, ewma_ucl, ewma_lcl, ewma_alerts = compute_ewma(x, target=robust_mean, sigma=robust_std)

    # Nelson / Western Electric trend rules
    nelson_alerts = detect_nelson_trend(x, centerline=robust_mean)

    # Multivariate Isolation Forest
    feat_df = pd.DataFrame(
        {
            "diameter_mm": df["diameter_mm"],
            "ambient_temp_c": df["ambient_temp_c"],
            "vibration_rms": df["vibration_rms"],
            "rolling_mean_3": df["diameter_mm"].rolling(3, min_periods=1).mean(),
            "rolling_std_3": df["diameter_mm"].rolling(3, min_periods=1).std().fillna(0.0),
        }
    )
    scaled = StandardScaler().fit_transform(feat_df)
    iso = IsolationForest(contamination=0.18, random_state=42)
    iso_preds = iso.fit_predict(scaled)
    iso_alerts = iso_preds == -1

    combined_alerts = limit_alerts | ewma_alerts | nelson_alerts | iso_alerts

    root_causes = []
    for idx, flagged in enumerate(combined_alerts):
        if flagged:
            root_causes.append(
                infer_root_cause(
                    df.iloc[idx],
                    robust_mean,
                    robust_std,
                    bool(ewma_alerts[idx]),
                    bool(nelson_alerts[idx]),
                )
            )
        else:
            root_causes.append("Normal")

    return {
        "robust_mean": robust_mean,
        "robust_std": robust_std,
        "ucl": robust_ucl,
        "lcl": robust_lcl,
        "ewma_z": ewma_z,
        "ewma_ucl": ewma_ucl,
        "ewma_lcl": ewma_lcl,
        "limit_alerts": limit_alerts,
        "ewma_alerts": ewma_alerts,
        "nelson_alerts": nelson_alerts,
        "iso_alerts": iso_alerts,
        "alerts": combined_alerts,
        "root_causes": root_causes,
    }


def evaluate_detection(y_true: np.ndarray, y_pred: np.ndarray, drift_start_idx: int = 16) -> dict:
    """Compute detection metrics and early-warning lead time on the drift segment."""
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    drift_indices = np.where(y_pred[drift_start_idx:])[0]
    first_drift_sample = int(drift_start_idx + drift_indices[0] + 1) if len(drift_indices) > 0 else None
    delay_samples = (first_drift_sample - (drift_start_idx + 1)) if first_drift_sample is not None else None

    return {
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1_score": round(f1, 3),
        "flagged_count": int(np.sum(y_pred)),
        "first_drift_alert_sample": first_drift_sample,
        "drift_detection_delay_samples": delay_samples,
    }


def plot_comparison(df: pd.DataFrame, pre: dict, post: dict, out_path: Path) -> None:
    """Generate Pre-AI vs. Post-AI SPC comparison figure."""
    s = df["sample_id"].to_numpy()
    x = df["diameter_mm"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # Panel 1: Pre-AI Classical SPC
    ax1 = axes[0]
    ax1.axvspan(17, 25, color="#fff3cd", alpha=0.6, label="Actual Tool-Wear Drift Zone (Samples 17–25)")
    ax1.plot(s, x, marker="o", color="#1f77b4", linewidth=1.8, label="Part Diameter (mm)")
    ax1.axhline(pre["mean"], color="green", linestyle="--", label=f"Sample Mean = {pre['mean']:.2f}")
    ax1.axhline(pre["ucl"], color="red", linestyle="--", label=f"UCL (+3σ) = {pre['ucl']:.2f}")
    ax1.axhline(pre["lcl"], color="red", linestyle="--", label=f"LCL (-3σ) = {pre['lcl']:.2f}")

    pre_idx = np.where(pre["alerts"])[0]
    if len(pre_idx) > 0:
        ax1.scatter(s[pre_idx], x[pre_idx], color="red", s=110, zorder=5, label="Pre-AI Alert (3σ Breach)")
    ax1.set_title("Pre-AI Optimization: Classical Static SPC (Outliers Inflate σ → Missed/Late Drift Detection)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Diameter (mm)")
    ax1.legend(loc="upper left", fontsize=8.5, ncol=2)
    ax1.grid(alpha=0.3)

    # Panel 2: Post-AI Predictive SPC
    ax2 = axes[1]
    ax2.axvspan(17, 25, color="#fff3cd", alpha=0.6, label="Actual Tool-Wear Drift Zone (Samples 17–25)")
    ax2.plot(s, x, marker="o", color="#1f77b4", linewidth=1.8, label="Part Diameter (mm)")
    ax2.plot(s, post["ewma_z"], color="#9467bd", linewidth=2.0, linestyle="-.", label="EWMA Trend")
    ax2.axhline(post["robust_mean"], color="green", linestyle="--", label=f"Robust Median = {post['robust_mean']:.2f}")
    ax2.axhline(post["ucl"], color="red", linestyle="--", label=f"Robust UCL = {post['ucl']:.2f}")
    ax2.axhline(post["lcl"], color="red", linestyle="--", label=f"Robust LCL = {post['lcl']:.2f}")

    post_idx = np.where(post["alerts"])[0]
    if len(post_idx) > 0:
        ax2.scatter(s[post_idx], x[post_idx], color="#d62728", s=110, marker="D", zorder=5, label="Post-AI Alert (EWMA / Nelson / IsolationForest)")
    ax2.set_title("Post-AI Optimization: Robust Baseline + EWMA + Isolation Forest (Early Drift & Fault Detection)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Sample Number")
    ax2.set_ylabel("Diameter (mm)")
    ax2.set_xticks(s)
    ax2.legend(loc="upper left", fontsize=8.5, ncol=2)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "spc_measurements.csv"
    out_dir = base_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        generate_spc_dataset(seed=42).to_csv(data_path, index=False)

    df = pd.read_csv(data_path)
    y_true = (df["ground_truth_state"] != "In-Control").to_numpy()

    pre = run_pre_ai_spc(df)
    post = run_post_ai_spc(df)

    pre_metrics = evaluate_detection(y_true, pre["alerts"])
    post_metrics = evaluate_detection(y_true, post["alerts"])

    df["pre_ai_alert"] = pre["alerts"]
    df["post_ai_alert"] = post["alerts"]
    df["ai_inferred_root_cause"] = post["root_causes"]
    df.to_csv(out_dir / "spc_results_annotated.csv", index=False)

    plot_comparison(df, pre, post, out_dir / "spc_pre_vs_post_comparison.png")

    summary = {
        "pre_ai_spc": {
            "mean": round(pre["mean"], 3),
            "std": round(pre["std"], 3),
            "lcl": round(pre["lcl"], 3),
            "ucl": round(pre["ucl"], 3),
            **pre_metrics,
        },
        "post_ai_spc": {
            "robust_mean": round(post["robust_mean"], 3),
            "robust_std": round(post["robust_std"], 3),
            "lcl": round(post["lcl"], 3),
            "ucl": round(post["ucl"], 3),
            **post_metrics,
        },
    }
    with open(out_dir / "spc_metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=== PRE-AI SPC (Classical Static) ===")
    print(json.dumps(summary["pre_ai_spc"], indent=2))
    print("\n=== POST-AI SPC (Robust + EWMA + Nelson + Isolation Forest) ===")
    print(json.dumps(summary["post_ai_spc"], indent=2))
    print("\n=== Flagged Samples & AI Root-Cause Attribution ===")
    flagged = df[df["post_ai_alert"]][["sample_id", "diameter_mm", "ground_truth_state", "pre_ai_alert", "ai_inferred_root_cause"]]
    print(flagged.to_string(index=False))


if __name__ == "__main__":
    main()
