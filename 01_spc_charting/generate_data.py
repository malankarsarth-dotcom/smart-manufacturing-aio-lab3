"""Generate synthetic manufacturing process data for SPC Charting (Use Case 1).

Simulates 25 part diameter measurements (baseline mean=50.0 mm, std=1.5 mm)
along with contextual telemetry (ambient temperature, vibration, shift, operator)
and realistic process faults (raw material inconsistency, tool wear drift,
and machine calibration shift) for Pre-AI vs. Post-AI SPC comparison.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_spc_dataset(seed: int = 42) -> pd.DataFrame:
    """Create a 25-sample synthetic SPC dataset with labeled anomalies."""
    rng = np.random.default_rng(seed)
    n_samples = 25

    sample_ids = np.arange(1, n_samples + 1)
    timestamps = pd.date_range("2026-11-17 08:00", periods=n_samples, freq="15min")

    # Baseline normal process: Mean = 50.0, Std = 1.5
    diameters = rng.normal(loc=50.0, scale=1.5, size=n_samples)
    ambient_temp = rng.normal(loc=22.0, scale=0.6, size=n_samples)
    vibration = rng.normal(loc=0.50, scale=0.05, size=n_samples)

    operators = ["Op_A" if i < 12 else "Op_B" for i in range(n_samples)]
    shifts = ["Morning" if i < 16 else "Afternoon" for i in range(n_samples)]

    state = ["In-Control"] * n_samples
    root_cause = ["None"] * n_samples

    # Fault 1 (Sample 8): Sudden low outlier (raw material inconsistency)
    diameters[7] = 41.5
    state[7] = "Out-of-Control"
    root_cause[7] = "Raw material inconsistency (undersized stock)"

    # Fault 2 (Samples 17-25): Gradual tool wear + rising vibration/temperature
    # Samples 17-22 show subtle upward drift (inside naive 3-sigma, caught by AI/EWMA)
    # Samples 23-25 breach upper control limit (machine calibration / severe tool wear)
    drift_offsets = [1.8, 2.3, 2.7, 3.2, 3.8, 4.3, 5.1, 5.6, 6.2]
    for idx, offset in zip(range(16, 25), drift_offsets):
        diameters[idx] = 50.0 + offset + rng.normal(0, 0.25)
        vibration[idx] = 0.52 + 0.04 * (idx - 15) + rng.normal(0, 0.01)
        ambient_temp[idx] = 22.5 + 0.35 * (idx - 15) + rng.normal(0, 0.15)
        state[idx] = "Drift-Warning" if idx < 22 else "Out-of-Control"
        root_cause[idx] = (
            "Gradual tool wear & thermal drift"
            if idx < 22
            else "Severe tool wear / Machine calibration shift"
        )

    df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M"),
            "diameter_mm": np.round(diameters, 3),
            "ambient_temp_c": np.round(ambient_temp, 2),
            "vibration_rms": np.round(vibration, 3),
            "operator": operators,
            "shift": shifts,
            "ground_truth_state": state,
            "true_root_cause": root_cause,
        }
    )
    return df


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = generate_spc_dataset(seed=42)
    out_path = data_dir / "spc_measurements.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved synthetic SPC dataset ({len(df)} rows) to: {out_path}")
    print(df[["sample_id", "diameter_mm", "ambient_temp_c", "vibration_rms", "ground_truth_state"]].to_string(index=False))


if __name__ == "__main__":
    main()
