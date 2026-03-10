from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "fx"
OUT_DIR = ROOT / "outputs" / "validation_fx"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_fx_series(
    path: Path | None = None,
    *,
    expect_pair: tuple[str, str] = ("JPY", "USD"),
) -> pd.DataFrame:
    if path is None:
        # default file
        p_parq = DATA_DIR / "JPYUSD_oanda_daily.parquet"
        p_csv = DATA_DIR / "JPYUSD_oanda_daily.csv"
        path = p_parq if p_parq.exists() else p_csv

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    df["obs_date"] = pd.to_datetime(df["obs_date"]).dt.normalize()
    df = df.sort_values("obs_date").drop_duplicates(subset=["obs_date"], keep="last").reset_index(drop=True)

    base, quote = expect_pair
    if not ((df["base_ccy"].astype(str).str.upper() == base) & (df["quote_ccy"].astype(str).str.upper() == quote)).all():
        # allow mixed if user merged; just warn by raising explicit error
        raise ValueError(f"Expected pair {base}{quote} in columns base_ccy/quote_ccy.")

    df["spot"] = pd.to_numeric(df["spot"], errors="coerce")
    df = df.dropna(subset=["spot"]).reset_index(drop=True)
    return df

def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_spot"] = np.log(out["spot"].astype(float))
    out["r"] = out["log_spot"].diff()
    out = out.dropna(subset=["r"]).reset_index(drop=True)
    return out

def write_df(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

def safe_std(x: np.ndarray) -> float:
    s = np.std(x, ddof=1)
    return float(s) if s > 0 else float("nan")