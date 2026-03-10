from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.validation._common import load_fx_series, compute_log_returns, OUT_DIR, write_df

def main():
    df = load_fx_series()
    dfr = compute_log_returns(df)

    for w in (20, 60, 120, 252):
        dfr[f"rv_{w}d"] = dfr["r"].rolling(w).std(ddof=1)

    out_dir = OUT_DIR / "02_rolling_vol"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_df(dfr[["obs_date","r","rv_20d","rv_60d","rv_120d","rv_252d"]], out_dir / "rolling_vol.csv")

    # plot
    plt.figure()
    plt.plot(dfr["obs_date"], dfr["rv_20d"], label="20d")
    plt.plot(dfr["obs_date"], dfr["rv_60d"], label="60d")
    plt.plot(dfr["obs_date"], dfr["rv_252d"], label="252d")
    plt.legend()
    plt.title("JPYUSD Rolling Realized Volatility (std of log returns)")
    plt.xlabel("Date")
    plt.ylabel("Vol (daily)")
    plt.tight_layout()
    plt.savefig(out_dir / "rolling_vol.png", dpi=180)
    plt.close()

    print(f"Wrote {out_dir}")

if __name__ == "__main__":
    main()