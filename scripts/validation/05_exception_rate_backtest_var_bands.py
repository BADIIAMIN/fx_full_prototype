from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scripts.validation._common import load_fx_series, compute_log_returns, OUT_DIR, write_df

def main():
    df = load_fx_series()
    dfr = compute_log_returns(df)

    w = 60
    alpha = 0.01  # 99% VaR (one-sided on loss)
    z_alpha = stats.norm.ppf(alpha)  # negative

    sig = dfr["r"].rolling(w).std(ddof=1)
    mu = dfr["r"].rolling(w).mean()

    # predicted left-tail VaR for returns
    var = (mu + z_alpha * sig)
    out = dfr[["obs_date", "r"]].copy()
    out["mu_hat"] = mu
    out["sig_hat"] = sig
    out["var_99"] = var

    out = out.dropna().reset_index(drop=True)
    out["breach"] = (out["r"] < out["var_99"]).astype(int)

    n = len(out)
    breaches = int(out["breach"].sum())
    expected = alpha * n

    # simple binomial test (two-sided)
    pval = stats.binomtest(breaches, n, alpha).pvalue if n > 0 else float("nan")

    summary = pd.DataFrame([{
        "window": w,
        "alpha": alpha,
        "n": n,
        "breaches": breaches,
        "expected": expected,
        "breach_rate": breaches / n if n else float("nan"),
        "binom_pvalue": float(pval),
    }])

    out_dir = OUT_DIR / "05_var_backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_df(summary, out_dir / "summary.csv")
    write_df(out, out_dir / "series.csv")

    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()