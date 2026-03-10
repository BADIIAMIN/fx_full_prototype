from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scripts.validation._common import load_fx_series, compute_log_returns, OUT_DIR, write_df

def main():
    df = load_fx_series()
    dfr = compute_log_returns(df)

    dfr["rv_20"] = dfr["r"].rolling(20).std(ddof=1)
    dfr["rv_60"] = dfr["r"].rolling(60).std(ddof=1)

    feat = dfr[["rv_20","rv_60"]].dropna()
    X = feat.to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    g = 3
    gmm = GaussianMixture(n_components=g, covariance_type="full", random_state=42)
    labels = gmm.fit_predict(Xs)
    probs = gmm.predict_proba(Xs)

    out = dfr.loc[feat.index, ["obs_date","spot","r","rv_20","rv_60"]].copy()
    out["regime"] = labels
    for j in range(g):
        out[f"p_regime_{j}"] = probs[:, j]

    out_dir = OUT_DIR / "07_clustering_gmm"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_df(out, out_dir / "regimes.csv")

    print(f"Wrote {out_dir}")

if __name__ == "__main__":
    main()