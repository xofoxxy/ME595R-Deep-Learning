"""
bayes_hackers_diet_ax.py

Simple "Hacker's Diet" trend estimator using a Bayesian state-space model
(Kalman filter) where we tune the noise hyperparameters with Meta's Ax.

Input CSV: must contain columns:
  - date (parseable by pandas)
  - weight (float)

Outputs:
  - <input>_trend.csv with columns:
      date, weight, trend_mean, trend_sd, one_step_pred, one_step_pred_sd

Install:
  pip install ax-platform pandas numpy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ax.service.ax_client import AxClient


# ----------------------------
# Kalman filter (local level model)
# x_t = x_{t-1} + v_t,  v_t ~ N(0, q)
# y_t = x_t + e_t,      e_t ~ N(0, r)
# ----------------------------

@dataclass
class KFResult:
    trend_mean: np.ndarray
    trend_var: np.ndarray
    one_step_pred: np.ndarray
    one_step_pred_var: np.ndarray


def kalman_filter_local_level(y: np.ndarray, q: float, r: float) -> KFResult:
    """
    Returns filtered posterior (trend_mean/var) and one-step-ahead prediction.
    """
    n = len(y)
    trend_mean = np.zeros(n, dtype=float)
    trend_var = np.zeros(n, dtype=float)
    one_step_pred = np.zeros(n, dtype=float)
    one_step_pred_var = np.zeros(n, dtype=float)

    # Initialize prior
    x = float(y[0])   # prior mean
    P = 10.0          # prior variance (broad)

    for t in range(n):
        # Predict
        x_pred = x
        P_pred = P + q

        one_step_pred[t] = x_pred
        one_step_pred_var[t] = P_pred + r

        # Update
        K = P_pred / (P_pred + r)
        x = x_pred + K * (y[t] - x_pred)
        P = (1.0 - K) * P_pred

        trend_mean[t] = x
        trend_var[t] = P

    return KFResult(
        trend_mean=trend_mean,
        trend_var=trend_var,
        one_step_pred=one_step_pred,
        one_step_pred_var=one_step_pred_var,
    )


def train_valid_split(y: np.ndarray, valid_frac: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple contiguous split: first (1-valid_frac) for training, remainder for validation.
    """
    n = len(y)
    n_valid = max(3, int(round(n * valid_frac)))
    n_train = max(3, n - n_valid)
    if n_train + n_valid > n:
        n_valid = n - n_train
    return y[:n_train], y[n_train:]


def evaluate_params(y_train: np.ndarray, y_valid: np.ndarray, q: float, r: float) -> float:
    """
    Validation score: one-step-ahead predictive Gaussian NLL on the validation segment.
    Filtering is run on the full series (train+valid) but we only score the valid part.
    """
    y_all = np.concatenate([y_train, y_valid])
    res = kalman_filter_local_level(y_all, q=q, r=r)

    start = len(y_train)
    mu = res.one_step_pred[start:]
    s2 = np.maximum(res.one_step_pred_var[start:], 1e-9)
    yv = y_valid

    nll = 0.5 * (np.log(2.0 * np.pi * s2) + ((yv - mu) ** 2) / s2)
    return float(np.sum(nll))


def tune_with_ax(y: np.ndarray, trials: int = 30, valid_frac: float = 0.25, seed: int = 0) -> tuple[float, float]:
    """
    Uses Ax to optimize q and r (process/measurement variances) to minimize validation NLL.
    """
    y_train, y_valid = train_valid_split(y, valid_frac=valid_frac)

    ax = AxClient(random_seed=seed, verbose_logging=False)
    ax.create_experiment(
        name="hd_kalman_qr_tuning",
        parameters=[
            {
                "name": "log10_q",
                "type": "range",
                "bounds": [-6.0, 0.5],
                "value_type": "float",
            },
            {
                "name": "log10_r",
                "type": "range",
                "bounds": [-4.0, 1.5],
                "value_type": "float",
            },
        ],
        objectives={
            "val_nll": {"minimize": True}
        },
    )

    for _ in range(trials):
        params, trial_index = ax.get_next_trial()
        q = 10.0 ** float(params["log10_q"])
        r = 10.0 ** float(params["log10_r"])

        # Guards
        q = float(np.clip(q, 1e-9, 1e3))
        r = float(np.clip(r, 1e-9, 1e3))

        val_nll = evaluate_params(y_train, y_valid, q=q, r=r)
        ax.complete_trial(trial_index=trial_index, raw_data=val_nll)

    best_params, _ = ax.get_best_parameters()
    q_best = float(np.clip(10.0 ** float(best_params["log10_q"]), 1e-9, 1e3))
    r_best = float(np.clip(10.0 ** float(best_params["log10_r"]), 1e-9, 1e3))
    return q_best, r_best


def load_series(csv_path: Path) -> pd.DataFrame:
    """
    Loads CSV and returns a clean daily time series with columns:
      - date (datetime64[ns])
      - weight (float)

    Accepts headers like Date/Weight, date/weight, with extra whitespace, etc.
    Averages multiple measurements within the same calendar day.
    """
    df = pd.read_csv(csv_path)

    # Normalize headers (fixes Date vs date vs " date ")
    df.columns = df.columns.str.strip().str.lower()

    if "date" not in df.columns or "weight" not in df.columns:
        raise ValueError(
            f"CSV must contain columns 'date' and 'weight'. Found: {df.columns.tolist()}"
        )

    # Parse types robustly
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    # Drop bad rows, sort
    df = df.dropna(subset=["date", "weight"]).sort_values("date").reset_index(drop=True)

    # Make day a real column (avoids pandas groupby warning / future behavior changes)
    df["day"] = df["date"].dt.floor("D")
    df = df.groupby("day", as_index=False)["weight"].mean()
    df = df.rename(columns={"day": "date"})
    df["date"] = pd.to_datetime(df["date"])

    if len(df) < 8:
        raise ValueError("Need at least ~8 rows for tuning to be meaningful.")

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=str, help="Path to CSV with columns date, weight")
    ap.add_argument("--trials", type=int, default=30, help="Ax trials (default: 30)")
    ap.add_argument("--valid-frac", type=float, default=0.25, help="Validation fraction (default: 0.25)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    args = ap.parse_args()

    # IMPORTANT: use the CLI argument (fixes your earlier bug)
    csv_path = Path(args.csv)

    df = load_series(csv_path)
    y = df["weight"].to_numpy(dtype=float)

    q_best, r_best = tune_with_ax(y, trials=args.trials, valid_frac=args.valid_frac, seed=args.seed)

    res = kalman_filter_local_level(y, q=q_best, r=r_best)
    df_out = df.copy()
    df_out["trend_mean"] = res.trend_mean
    df_out["trend_sd"] = np.sqrt(np.maximum(res.trend_var, 0.0))
    df_out["one_step_pred"] = res.one_step_pred
    df_out["one_step_pred_sd"] = np.sqrt(np.maximum(res.one_step_pred_var, 0.0))

    out_path = csv_path.with_name(csv_path.stem + "_trend.csv")
    df_out.to_csv(out_path, index=False)

    print(f"Best q (process variance):     {q_best:.6g}")
    print(f"Best r (measurement variance): {r_best:.6g}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
