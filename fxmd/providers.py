# fxmd/providers.py
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Protocol

import pandas as pd
import requests

from dotenv import load_dotenv
import os

load_dotenv()

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
OANDA_KEY = os.getenv("OANDA_API_KEY")
POLYGON_KEY = os.getenv("POLYGON_API_KEY")


# -----------------------
# Shared utilities
# -----------------------

def _to_unix(d: date) -> int:
    # interpret as start-of-day UTC
    dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return int(dt.timestamp())

def _ensure_sorted_unique(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("obs_date").drop_duplicates(subset=["obs_date"], keep="last")
    return df.reset_index(drop=True)

def _retry_get_json(url: str, params: dict, headers: Optional[dict] = None,
                    retries: int = 3, backoff: float = 1.5, timeout: float = 30.0) -> dict:
    last_exc = None
    for k in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            time.sleep(backoff ** k)
    raise RuntimeError(f"HTTP request failed after {retries} attempts: {last_exc}") from last_exc


# -----------------------
# Provider interface
# -----------------------

class FxTimeSeriesProvider(Protocol):
    def fetch_spot_series(
        self,
        base_ccy: str,
        quote_ccy: str,
        start: date,
        end: date,
        *,
        curve_type: str = "Exchange",
    ) -> pd.DataFrame:
        """
        Returns canonical dataframe with columns:
          curve_type, base_ccy, obs_date, quote_ccy, spot
        obs_date is a pandas Timestamp (date).
        """


# -----------------------
# Alpha Vantage provider
# -----------------------

@dataclass(frozen=True)
class AlphaVantageFxProvider:
    api_key: str
    outputsize: Literal["compact", "full"] = "full"
    session: Optional[requests.Session] = None

    def fetch_spot_series(
        self,
        base_ccy: str,
        quote_ccy: str,
        start: date,
        end: date,
        *,
        curve_type: str = "Exchange",
    ) -> pd.DataFrame:
        """
        Uses Alpha Vantage FX_DAILY (or similar) style endpoint.
        Docs: Alpha Vantage documentation. :contentReference[oaicite:4]{index=4}
        """
        base_ccy = base_ccy.upper().strip()
        quote_ccy = quote_ccy.upper().strip()

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "FX_DAILY",
            "from_symbol": base_ccy,
            "to_symbol": quote_ccy,
            "outputsize": self.outputsize,
            "datatype": "json",
            "apikey": self.api_key,
        }

        js = _retry_get_json(url, params=params)

        # Error handling (Alpha Vantage often returns {"Error Message":...} or {"Note":...})
        if "Error Message" in js:
            raise RuntimeError(f"Alpha Vantage error: {js['Error Message']}")
        if "Note" in js:
            raise RuntimeError(f"Alpha Vantage throttling/note: {js['Note']}")

        # FX_DAILY typically stores data under a time-series key.
        # We'll look for a key that contains "Time Series".
        ts_key = next((k for k in js.keys() if "time series" in k.lower()), None)
        if ts_key is None:
            raise RuntimeError(f"Alpha Vantage response missing time series key. Keys: {list(js.keys())}")

        ts = js[ts_key]  # dict: { "YYYY-MM-DD": {"1. open":..., "4. close":...} ... }

        rows = []
        for d_str, ohlc in ts.items():
            obs = pd.to_datetime(d_str, errors="coerce").normalize()
            if pd.isna(obs):
                continue
            obs_date = obs.date()
            if obs_date < start or obs_date > end:
                continue

            # For a "spot" time series, we typically take close.
            close = None
            for cand_key in ("4. close", "5. adjusted close", "close"):
                if cand_key in ohlc:
                    close = ohlc[cand_key]
                    break
            if close is None:
                # fallback: any numeric field
                close = next(iter(ohlc.values()))

            rows.append(
                {
                    "curve_type": curve_type,
                    "base_ccy": base_ccy,
                    "obs_date": pd.Timestamp(obs_date),
                    "quote_ccy": quote_ccy,
                    "spot": float(close),
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("Alpha Vantage returned no rows in the requested date range.")
        return _ensure_sorted_unique(df)


# -----------------------
# Finnhub provider
# -----------------------

@dataclass(frozen=True)
class FinnhubFxProvider:
    api_key: str
    # Finnhub expects symbols like "OANDA:USD_JPY" for forex candles. :contentReference[oaicite:5]{index=5}
    venue: str = "OANDA"
    session: Optional[requests.Session] = None

    def _symbol(self, base_ccy: str, quote_ccy: str) -> str:
        base_ccy = base_ccy.upper().strip()
        quote_ccy = quote_ccy.upper().strip()
        return f"{self.venue}:{base_ccy}_{quote_ccy}"

    def fetch_spot_series(
        self,
        base_ccy: str,
        quote_ccy: str,
        start: date,
        end: date,
        *,
        curve_type: str = "Exchange",
        resolution: Literal["1", "5", "15", "30", "60", "D", "W", "M"] = "D",
    ) -> pd.DataFrame:
        """
        Uses Finnhub forex candles endpoint. :contentReference[oaicite:6]{index=6}
        """
        sym = self._symbol(base_ccy, quote_ccy)

        url = "https://finnhub.io/api/v1/forex/candle"
        params = {
            "symbol": sym,
            "resolution": resolution,
            "from": _to_unix(start),
            # Finnhub 'to' is inclusive-ish; use end+1 day start-of-day to be safe
            "to": _to_unix(end) + 24 * 3600 - 1,
            "token": self.api_key,
        }

        js = _retry_get_json(url, params=params)

        # Finnhub returns { "s": "ok", "t": [...], "c": [...], ... } or error payload.
        if "error" in js:
            raise RuntimeError(f"Finnhub error: {js['error']}")  # e.g. access denied :contentReference[oaicite:7]{index=7}
        if js.get("s") != "ok":
            raise RuntimeError(f"Finnhub candle status not ok: {js}")

        t = js.get("t", [])
        c = js.get("c", [])
        if not t or not c or len(t) != len(c):
            raise RuntimeError(f"Finnhub candle payload invalid/empty for symbol={sym}: {js}")

        rows = []
        for ts, close in zip(t, c):
            # ts is unix seconds
            obs_dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
            if obs_dt < start or obs_dt > end:
                continue
            rows.append(
                {
                    "curve_type": curve_type,
                    "base_ccy": base_ccy.upper().strip(),
                    "obs_date": pd.Timestamp(obs_dt),
                    "quote_ccy": quote_ccy.upper().strip(),
                    "spot": float(close),
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError(f"Finnhub returned no rows in the requested date range for {sym}.")
        return _ensure_sorted_unique(df)


# -----------------------
# Writers: match your file shape
# -----------------------

def write_fx_spot_headerless_csv(df: pd.DataFrame, path: str | Path, date_format: str = "%d/%m/%Y") -> None:
    """
    Writes rows in the same headerless format as your sample:
      Exchange, USD, 02/02/2026, JPY, 155.53
    """
    path = Path(path)
    df = df.copy()
    df["obs_date"] = pd.to_datetime(df["obs_date"]).dt.date

    with path.open("w", newline="", encoding="utf-8") as f:
        w = __import__("csv").writer(f)
        for _, r in df.iterrows():
            w.writerow(
                [
                    r["curve_type"],
                    r["base_ccy"],
                    r["obs_date"].strftime(date_format),
                    r["quote_ccy"],
                    f"{float(r['spot']):.2f}",
                ]
            )