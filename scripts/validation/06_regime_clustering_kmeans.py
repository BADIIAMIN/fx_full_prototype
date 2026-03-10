from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scripts.validation._common import load_fx_series, compute_log_returns, OUT_DIR, write_df

def max_drawdown(x: np.ndarray) -> float:
    # x is cumulative log return
    peak = np.maximum.accumulate(x)
    dd = x - peak
    return float(dd.min())

def main():
    df = load_fx_series()
    dfr = compute_log_returns(df)

    # Features
    dfr["rv_20"] = dfr["r"].rolling(20).std(ddof=1)
    dfr["rv_60"] = dfr["r"].rolling(60).std(ddof=1)
    dfr["cum_lr"] = dfr["r"].cumsum()

    # rolling drawdown proxy
    # compute rolling max drawdown over 252d using cumulative log-returns
    window = 252
    dd = []
    cl = dfr["cum_lr"].to_numpy()
    for i in range(len(cl)):
        if i < window:
            dd.append(np.nan)
        else:
            seg = cl[i-window:i+1]
            dd.append(max_drawdown(seg))
    dfr["mdd_252"] = dd

    feat = dfr[["rv_20","rv_60","mdd_252"]].dropna()
    X = feat.to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    k = 3
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(Xs)

    out = dfr.loc[feat.index, ["obs_date","spot","r","rv_20","rv_60","mdd_252"]].copy()
    out["cluster"] = labels

    out_dir = OUT_DIR / "06_clustering_kmeans"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_df(out, out_dir / "clusters.csv")

    # cluster summary
    summ = out.groupby("cluster")[["rv_20","rv_60","mdd_252","r"]].agg(["mean","std","min","max"])
    summ.to_csv(out_dir / "cluster_summary.csv")

    print(f"Wrote {out_dir}")

if __name__ == "__main__":
    main()