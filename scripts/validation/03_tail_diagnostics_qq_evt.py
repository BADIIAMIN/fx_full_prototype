from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scripts.validation._common import load_fx_series, compute_log_returns, OUT_DIR, write_df, safe_std

def excess_kurtosis(x: np.ndarray) -> float:
    return float(stats.kurtosis(x, fisher=True, bias=False))

def main():
    df = load_fx_series()
    dfr = compute_log_returns(df)
    r = dfr["r"].to_numpy()

    out_dir = OUT_DIR / "03_tail"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Basic tail stats
    ek = excess_kurtosis(r)
    skew = float(stats.skew(r, bias=False))
    write_df(pd.DataFrame([{"excess_kurtosis": ek, "skewness": skew}]), out_dir / "tail_stats.csv")

    # QQ plot vs Normal
    plt.figure()
    stats.probplot(r, dist="norm", plot=plt)
    plt.title("QQ Plot of JPYUSD log-returns vs Normal")
    plt.tight_layout()
    plt.savefig(out_dir / "qqplot_normal.png", dpi=180)
    plt.close()

    # EVT threshold scan: mean excess over threshold (upper tail on |r|)
    absr = np.abs(r)
    qs = np.linspace(0.90, 0.995, 20)
    rows = []
    for q in qs:
        u = np.quantile(absr, q)
        exceed = absr[absr > u] - u
        me = float(np.mean(exceed)) if len(exceed) > 5 else float("nan")
        rows.append({"quantile": float(q), "threshold_u": float(u), "n_exceed": int(len(exceed)), "mean_excess": me})

    evt = pd.DataFrame(rows)
    write_df(evt, out_dir / "evt_threshold_scan.csv")

    plt.figure()
    plt.plot(evt["threshold_u"], evt["mean_excess"])
    plt.title("Mean Excess Function (|r| tail)")
    plt.xlabel("Threshold u")
    plt.ylabel("Mean excess E[|r|-u | |r|>u]")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_excess.png", dpi=180)
    plt.close()

    print(f"Wrote {out_dir}")

if __name__ == "__main__":
    main()