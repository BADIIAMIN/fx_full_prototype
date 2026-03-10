from __future__ import annotations

import os
import time
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Literal

import pandas as pd
import requests
from dotenv import load_dotenv


# -----------------------
# Helpers
# -----------------------

def _to_rfc3339(d: date) -> str:
    # start of day UTC
    dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")

def _ensure_sorted_unique(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("obs_date").drop_duplicates(subset=["obs_date"], keep="last")
    return df.reset_index(drop=True)

def _retry_get_json(url: str, params: dict, headers: dict, retries: int = 3, backoff: float = 1.7, timeout: float = 30.0) -> dict:
    last = None
    for k in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(backoff ** k)
    raise RuntimeError(f"HTTP request failed after {retries} attempts: {last}") from last

def write_fx_spot_headerless_csv(df: pd.DataFrame, path: str | Path, date_format: str = "%d/%m/%Y") -> None:
    """
    Writes headerless rows:
      Exchange, JPY, 02/02/2026, USD, 0.006423
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["obs_date"] = pd.to_datetime(df["obs_date"]).dt.date

    import csv
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for _, r in df.iterrows():
            w.writerow([
                r["curve_type"],
                r["base_ccy"],
                r["obs_date"].strftime(date_format),
                r["quote_ccy"],
                f"{float(r['spot']):.10f}",
            ])

def qa_report(df: pd.DataFrame, name: str) -> None:
    df = df.sort_values("obs_date")
    obs = pd.to_datetime(df["obs_date"]).dt.date
    print(f"\n--- QA: {name} ---")
    print(f"rows={len(df)}")
    print(f"start={obs.min()}  end={obs.max()}")
    print(f"min={df['spot'].min():.10f}  max={df['spot'].max():.10f}")
    # check duplicates
    dup = df.duplicated(subset=["obs_date"]).sum()
    print(f"duplicate_dates={dup}")
    # simple gap check (calendar-day gaps; FX trades 5d/week so don’t enforce daily)
    gaps = (pd.to_datetime(obs).diff().dt.days.dropna())
    if len(gaps) > 0:
        print(f"max_calendar_gap_days={int(gaps.max())}")


# -----------------------
# Provider: Oanda daily candles
# -----------------------

@dataclass(frozen=True)
class OandaFxDailyDownloader:
    api_key: str
    environment: Literal["practice", "live"] = "practice"
    price: Literal["M", "B", "A"] = "M"     # Mid/Bid/Ask

    def _host(self) -> str:
        return "https://api-fxpractice.oanda.com" if self.environment == "practice" else "https://api-fxtrade.oanda.com"

    def fetch_usdjpy_daily(self, start: date, end: date) -> pd.DataFrame:
        """
        Fetch USD_JPY daily candles from Oanda between start and end (inclusive).
        Oanda may limit number of candles returned per request; we chunk by time.
        """
        instrument = "USD_JPY"
        url = f"{self._host()}/v3/instruments/{instrument}/candles"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # conservative chunk: ~1500 days per call to stay away from limits
        chunk_days = 1500

        rows = []
        cur = start
        while cur <= end:
            cur_end = min(end, cur + timedelta(days=chunk_days - 1))

            params = {
                "price": self.price,
                "granularity": "D",
                "from": _to_rfc3339(cur),
                # Oanda 'to' acts like an upper bound; include next day start to be safe
                "to": _to_rfc3339(cur_end + timedelta(days=1)),
            }

            js = _retry_get_json(url, params=params, headers=headers)

            candles = js.get("candles", [])
            if not candles:
                if "errorMessage" in js:
                    raise RuntimeError(f"Oanda error: {js['errorMessage']}")
                # If empty chunk, just move on
                cur = cur_end + timedelta(days=1)
                continue

            for c in candles:
                if not c.get("complete", True):
                    continue
                t = c.get("time")
                if not t:
                    continue
                obs_d = pd.to_datetime(t, errors="coerce").date()
                if obs_d < start or obs_d > end:
                    continue

                px = c.get({"M": "mid", "B": "bid", "A": "ask"}[self.price], {})
                close = px.get("c")
                if close is None:
                    continue

                rows.append({
                    "curve_type": "Exchange",
                    "base_ccy": "USD",
                    "quote_ccy": "JPY",
                    "obs_date": pd.Timestamp(obs_d),
                    "spot": float(close),
                })

            # be gentle to API
            time.sleep(0.15)
            cur = cur_end + timedelta(days=1)

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("No USDJPY candles returned from Oanda for the requested window.")
        return _ensure_sorted_unique(df)


# -----------------------
# Transform: USDJPY -> JPYUSD
# -----------------------

def invert_pair(df_usdjpy: pd.DataFrame) -> pd.DataFrame:
    df = df_usdjpy.copy()
    if not ((df["base_ccy"] == "USD") & (df["quote_ccy"] == "JPY")).all():
        raise ValueError("Expected USDJPY dataframe for inversion.")
    df["spot"] = 1.0 / df["spot"].astype(float)
    df["base_ccy"] = "JPY"
    df["quote_ccy"] = "USD"
    return df


# -----------------------
# Main
# -----------------------

def main():
    load_dotenv()
    oanda_key = os.getenv("OANDA_API_KEY")
    if not oanda_key:
        raise RuntimeError("Missing OANDA_API_KEY in .env")

    # choose a long backtesting window
    # example: 2005-01-03 to 2026-03-01
    start = date(2005, 1, 3)
    end = date(2026, 3, 1)

    out_dir = Path(__file__).resolve().parent.parent / "data" / "fx"
    out_dir.mkdir(parents=True, exist_ok=True)

    dl = OandaFxDailyDownloader(api_key=oanda_key, environment="practice", price="M")

    usdjpy = dl.fetch_usdjpy_daily(start, end)
    qa_report(usdjpy, "USDJPY (Oanda, daily, mid)")

    jpyusd = invert_pair(usdjpy)
    qa_report(jpyusd, "JPYUSD (inverted from USDJPY)")

    # Save canonical formats
    usdjpy.to_parquet(out_dir / "USDJPY_oanda_daily.parquet", index=False)
    jpyusd.to_parquet(out_dir / "JPYUSD_oanda_daily.parquet", index=False)

    usdjpy.to_csv(out_dir / "USDJPY_oanda_daily.csv", index=False)
    jpyusd.to_csv(out_dir / "JPYUSD_oanda_daily.csv", index=False)

    # Save in YOUR headerless template format for your parser
    write_fx_spot_headerless_csv(jpyusd, out_dir / "JPYUSD_timeseries_headerless.csv")
    print(f"\nWrote:\n- {out_dir / 'JPYUSD_oanda_daily.parquet'}\n- {out_dir / 'JPYUSD_timeseries_headerless.csv'}")


if __name__ == "__main__":
    main()