# fx_vol_clustering_tests.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import acf

import matplotlib.pyplot as plt


# ----------------------------
# Core utilities
# ----------------------------

def load_fx_spot_csv(
    path: str | Path,
    date_col: str = "date",
    spot_col: str = "spot",
    tz: Optional[str] = None,
) -> pd.Series:
    """
    Load FX spot time series from a CSV with columns [date_col, spot_col].
    Returns a pandas Series indexed by datetime.
    """
    df = pd.read_csv(path)
    if date_col not in df.columns or spot_col not in df.columns:
        raise ValueError(f"CSV must contain columns: {date_col}, {spot_col}")
    dt = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    if dt.isna().any():
        raise ValueError("Some dates could not be parsed.")
    if tz:
        dt = dt.dt.tz_convert(tz)
    s = pd.Series(df[spot_col].astype(float).values, index=dt, name=spot_col)
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if (s <= 0).any():
        raise ValueError("Spot must be strictly positive.")
    return s


def compute_log_returns(
    spot: pd.Series,
    freq: Optional[str] = None,
    dropna: bool = True
) -> pd.Series:
    """
    Compute log returns. Optionally resample to a given frequency (e.g. 'B', 'D', 'W').
    """
    s = spot.copy()
    if freq is not None:
        # Use last observation in each period; adjust to your convention if needed
        s = s.resample(freq).last().dropna()
    r = np.log(s).diff()
    r.name = "log_return"
    return r.dropna() if dropna else r


def annualised_vol_from_returns(r: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualised volatility from log returns (simple estimator).
    """
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def simulate_gbm_log_returns(
    n: int,
    sigma_ann: float,
    dt: float = 1.0/252.0,
    mu_ann: float = 0.0,
    seed: Optional[int] = 1234
) -> np.ndarray:
    """
    Simulate GBM log returns with constant sigma.
    r_t = (mu - 0.5 sigma^2) dt + sigma sqrt(dt) eps
    where sigma and mu are annualised.
    """
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    drift = (mu_ann - 0.5 * sigma_ann**2) * dt
    diff = sigma_ann * np.sqrt(dt) * eps
    return drift + diff


# ----------------------------
# Tests & outputs
# ----------------------------

@dataclass
class VolClusteringResults:
    n_obs: int
    freq_label: str
    annualised_vol: float
    kurtosis_excess: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    ljungbox_lags: int
    ljungbox_stat: float
    ljungbox_pvalue: float
    arch_lags: int
    arch_lm_stat: float
    arch_lm_pvalue: float
    acf_sq_first10: List[float]  # first 10 lags ACF of r^2 (lag 1..10)


def run_volatility_clustering_suite(
    r: pd.Series,
    freq_label: str = "raw",
    ljungbox_lags: int = 20,
    arch_lags: int = 10,
    periods_per_year: int = 252
) -> VolClusteringResults:
    """
    Runs:
      - Jarque-Bera on returns
      - Ljung-Box on squared returns
      - Engle ARCH LM test
      - ACF of squared returns (first 10 lags)
    """
    r = r.dropna().astype(float)
    if len(r) < max(ljungbox_lags, arch_lags) + 20:
        raise ValueError("Not enough observations to run tests reliably.")

    # Basic stats
    vol_ann = annualised_vol_from_returns(r, periods_per_year=periods_per_year)
    kurt_excess = float(pd.Series(r).kurtosis(fisher=True))  # excess kurtosis

    jb_stat, jb_p, _, _ = jarque_bera(r.values)

    # Ljung-Box on squared returns
    r2 = (r.values ** 2)
    lb = acorr_ljungbox(r2, lags=[ljungbox_lags], return_df=True)
    lb_stat = float(lb["lb_stat"].iloc[-1])
    lb_p = float(lb["lb_pvalue"].iloc[-1])

    # Engle ARCH LM test
    arch_stat, arch_p, _, _ = het_arch(r.values, nlags=arch_lags)

    # ACF of squared returns (exclude lag 0)
    acf_vals = acf(r2, nlags=10, fft=True)
    acf_first10 = [float(x) for x in acf_vals[1:11]]

    return VolClusteringResults(
        n_obs=int(len(r)),
        freq_label=freq_label,
        annualised_vol=float(vol_ann),
        kurtosis_excess=float(kurt_excess),
        jarque_bera_stat=float(jb_stat),
        jarque_bera_pvalue=float(jb_p),
        ljungbox_lags=int(ljungbox_lags),
        ljungbox_stat=float(lb_stat),
        ljungbox_pvalue=float(lb_p),
        arch_lags=int(arch_lags),
        arch_lm_stat=float(arch_stat),
        arch_lm_pvalue=float(arch_p),
        acf_sq_first10=acf_first10,
    )


# ----------------------------
# Plotting helpers (validation pack friendly)
# ----------------------------

def plot_returns_and_vol_proxy(
    r: pd.Series,
    outpath: str | Path,
    title_prefix: str = ""
) -> None:
    """
    Generates 2 plots:
      1) returns
      2) |returns| (volatility proxy) or returns^2
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    plt.plot(r.index, r.values)
    plt.title(f"{title_prefix} FX log returns")
    plt.xlabel("Date")
    plt.ylabel("log-return")
    plt.tight_layout()
    fig.savefig(outpath.with_suffix(".returns.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(r.index, np.abs(r.values))
    plt.title(f"{title_prefix} |FX log returns| (volatility proxy)")
    plt.xlabel("Date")
    plt.ylabel("|log-return|")
    plt.tight_layout()
    fig.savefig(outpath.with_suffix(".abs_returns.png"), dpi=150)
    plt.close(fig)


def plot_acf_squared_returns(
    r: pd.Series,
    nlags: int,
    outpath: str | Path,
    title_prefix: str = ""
) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    r2 = r.values**2
    acf_vals = acf(r2, nlags=nlags, fft=True)

    fig = plt.figure()
    plt.stem(range(0, nlags+1), acf_vals, use_line_collection=True)
    plt.title(f"{title_prefix} ACF of squared returns")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.tight_layout()
    fig.savefig(outpath.with_suffix(".acf_sq.png"), dpi=150)
    plt.close(fig)


# ----------------------------
# CLI-style entry point
# ----------------------------

def main(
    csv_path: str,
    date_col: str = "date",
    spot_col: str = "spot",
    resample_freq: Optional[str] = None,   # e.g. 'B' for business-daily, 'W' weekly
    periods_per_year: int = 252,
    ljungbox_lags: int = 20,
    arch_lags: int = 10,
    out_dir: str = "fx_vol_clustering_output",
    compare_with_gbm: bool = True,
    gbm_seed: int = 1234
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spot = load_fx_spot_csv(csv_path, date_col=date_col, spot_col=spot_col)
    r = compute_log_returns(spot, freq=resample_freq)

    freq_label = resample_freq or "raw"
    results_emp = run_volatility_clustering_suite(
        r, freq_label=freq_label,
        ljungbox_lags=ljungbox_lags, arch_lags=arch_lags,
        periods_per_year=periods_per_year
    )

    # plots
    plot_returns_and_vol_proxy(r, out_dir / f"emp_{freq_label}", title_prefix=f"[EMP {freq_label}]")
    plot_acf_squared_returns(r, nlags=40, outpath=out_dir / f"emp_{freq_label}", title_prefix=f"[EMP {freq_label}]")

    report: Dict[str, object] = {"empirical": asdict(results_emp)}

    # Optional: compare with GBM using empirical sigma (what GBM should look like)
    if compare_with_gbm:
        sigma_ann = results_emp.annualised_vol
        r_gbm = simulate_gbm_log_returns(
            n=len(r),
            sigma_ann=sigma_ann,
            dt=1.0/periods_per_year,
            mu_ann=0.0,
            seed=gbm_seed
        )
        r_gbm_s = pd.Series(r_gbm, index=r.index, name="gbm_log_return")

        results_gbm = run_volatility_clustering_suite(
            r_gbm_s, freq_label=f"gbm_calib_{freq_label}",
            ljungbox_lags=ljungbox_lags, arch_lags=arch_lags,
            periods_per_year=periods_per_year
        )

        plot_returns_and_vol_proxy(r_gbm_s, out_dir / f"gbm_{freq_label}", title_prefix=f"[GBM {freq_label}]")
        plot_acf_squared_returns(r_gbm_s, nlags=40, outpath=out_dir / f"gbm_{freq_label}", title_prefix=f"[GBM {freq_label}]")
        report["gbm_benchmark"] = asdict(results_gbm)

    # Save JSON report
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote outputs to: {out_dir.resolve()}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    # Example:
    # python fx_vol_clustering_tests.py -- (wrap with argparse if you want)
    # For quick use, just edit the call below or import main() elsewhere.
    pass
