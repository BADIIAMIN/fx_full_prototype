from __future__ import annotations

import numpy as np
import pandas as pd
from scripts.validation._common import load_fx_series, compute_log_returns, OUT_DIR, write_df

def main():
    df = load_fx_series()
    dfr = compute_log_returns(df)

    r = dfr["r"].to_numpy()
    stats = {
        "n": len(r),
        "start": str(dfr["obs_date"].min().date()),
        "end": str(dfr["obs_date"].max().date()),
        "mean_daily": float(np.mean(r)),
        "std_daily": float(np.std(r, ddof=1)),
        "min_daily": float(np.min(r)),
        "max_daily": float(np.max(r)),
    }
    out_stats = pd.DataFrame([stats])

    write_df(out_stats, OUT_DIR / "01_basic_stats" / "stats.csv")
    write_df(dfr[["obs_date","spot","r"]], OUT_DIR / "01_basic_stats" / "returns_series.csv")

    print(out_stats.to_string(index=False))

if __name__ == "__main__":
    main()