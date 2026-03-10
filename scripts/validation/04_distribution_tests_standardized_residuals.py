from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scripts.validation._common import load_fx_series, compute_log_returns, OUT_DIR, write_df

def main():
    df = load_fx_series()
    dfr = compute_log_returns(df)

    # standardize by rolling 60d vol (you can change)
    w = 60
    sig = dfr["r"].rolling(w).std(ddof=1)
    z = (dfr["r"] / sig).dropna().to_numpy()

    # KS vs N(0,1)
    ks = stats.kstest(z, "norm")

    # Anderson-Darling for normal
    ad = stats.anderson(z, dist="norm")

    # Cramer-von Mises vs normal
    cvm = stats.cramervonmises(z, stats.norm.cdf)

    out = {
        "n": len(z),
        "window_vol": w,
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "cvm_stat": float(cvm.statistic),
        "cvm_pvalue": float(cvm.pvalue),
        "ad_stat": float(ad.statistic),
        # AD critical values depend on significance levels
        "ad_crit_15pct": float(ad.critical_values[0]),
        "ad_crit_10pct": float(ad.critical_values[1]),
        "ad_crit_5pct": float(ad.critical_values[2]),
        "ad_crit_2_5pct": float(ad.critical_values[3]),
        "ad_crit_1pct": float(ad.critical_values[4]),
    }

    out_dir = OUT_DIR / "04_distribution_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_df(pd.DataFrame([out]), out_dir / "tests_summary.csv")
    pd.DataFrame({"z": z}).to_csv(out_dir / "standardized_residuals.csv", index=False)

    print(pd.DataFrame([out]).to_string(index=False))

if __name__ == "__main__":
    main()