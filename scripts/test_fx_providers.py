from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Literal, Optional, Dict, Any

import pandas as pd
import requests
from dotenv import load_dotenv


# -----------------------
# Shared utilities
# -----------------------

def _to_unix(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())

def _ensure_sorted_unique(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("obs_date").drop_duplicates(subset=["obs_date"], keep="last")
    return df.reset_index(drop=True)

def _retry_get_json(url: str, params: dict | None = None, headers: dict | None = None,
                    retries: int = 3, backoff: float = 1.6, timeout: float = 30.0) -> dict:
    last = None
    for k in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(backoff ** k)
    raise RuntimeError(f"HTTP failed after {retries} attempts: {last}") from last

def write_fx_spot_headerless_csv(df: pd.DataFrame, path: str | Path, date_format: str = "%d/%m/%Y") -> None:
    """
    Writes:
      Exchange, USD, 02/02/2026, JPY, 155.53
    """
    path = Path(path)
    df = df.copy()
    df["obs_date"] = pd.to_datetime(df["obs_date"]).dt.date
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for _, r in df.iterrows():
            w.writerow([
                r["curve_type"],
                r["base_ccy"],
                r["obs_date"].strftime(date_format),
                r["quote_ccy"],
                f"{float(r['spot']):.6f}",
            ])


# -----------------------
# Providers
# -----------------------

@dataclass(frozen=True)
class AlphaVantageFxProvider:
    api_key: str
    outputsize: Literal["compact", "full"] = "compact"

    def fetch_spot_series(self, base: str, quote: str, start: date, end: date, curve_type: str = "Exchange") -> pd.DataFrame:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "FX_DAILY",
            "from_symbol": base.upper(),
            "to_symbol": quote.upper(),
            "outputsize": self.outputsize,
            "datatype": "json",
            "apikey": self.api_key,
        }
        js = _retry_get_json(url, params=params)

        if "Error Message" in js:
            raise RuntimeError(f"AlphaVantage error: {js['Error Message']}")
        if "Note" in js:
            raise RuntimeError(f"AlphaVantage throttled: {js['Note']}")

        ts_key = next((k for k in js.keys() if "time series" in k.lower()), None)
        if not ts_key:
            raise RuntimeError(f"AlphaVantage missing time series key. Keys={list(js.keys())}")

        rows = []
        for d_str, ohlc in js[ts_key].items():
            obs = pd.to_datetime(d_str, errors="coerce")
            if pd.isna(obs):
                continue
            obs_d = obs.date()
            if obs_d < start or obs_d > end:
                continue
            close = ohlc.get("4. close") or ohlc.get("5. adjusted close") or next(iter(ohlc.values()))
            rows.append({
                "curve_type": curve_type,
                "base_ccy": base.upper(),
                "quote_ccy": quote.upper(),
                "obs_date": pd.Timestamp(obs_d),
                "spot": float(close),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("AlphaVantage returned no rows for requested range.")
        return _ensure_sorted_unique(df)


@dataclass(frozen=True)
class FinnhubFxProvider:
    api_key: str
    venue: str = "OANDA"  # symbol: OANDA:USD_JPY

    def fetch_spot_series(self, base: str, quote: str, start: date, end: date, curve_type: str = "Exchange",
                          resolution: Literal["1","5","15","30","60","D","W","M"] = "D") -> pd.DataFrame:
        sym = f"{self.venue}:{base.upper()}_{quote.upper()}"
        url = "https://finnhub.io/api/v1/forex/candle"
        params = {
            "symbol": sym,
            "resolution": resolution,
            "from": _to_unix(start),
            "to": _to_unix(end) + 24*3600 - 1,
            "token": self.api_key,
        }
        js = _retry_get_json(url, params=params)

        if "error" in js:
            raise RuntimeError(f"Finnhub error: {js['error']}")
        if js.get("s") != "ok":
            raise RuntimeError(f"Finnhub status not ok: {js}")

        t = js.get("t", [])
        c = js.get("c", [])
        if not t or not c or len(t) != len(c):
            raise RuntimeError(f"Finnhub empty/invalid candles: {js}")

        rows = []
        for ts, close in zip(t, c):
            obs_d = datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
            if obs_d < start or obs_d > end:
                continue
            rows.append({
                "curve_type": curve_type,
                "base_ccy": base.upper(),
                "quote_ccy": quote.upper(),
                "obs_date": pd.Timestamp(obs_d),
                "spot": float(close),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError(f"Finnhub returned no rows for {sym}.")
        return _ensure_sorted_unique(df)


@dataclass(frozen=True)
class OandaFxProvider:
    api_key: str
    environment: Literal["practice", "live"] = "practice"

    def fetch_spot_series(self, base: str, quote: str, start: date, end: date, curve_type: str = "Exchange",
                          granularity: str = "D") -> pd.DataFrame:
        """
        Oanda candles (most reliable). Requires instrument like USD_JPY.
        NOTE: Oanda also requires an account ID for trading endpoints,
        but historical candles endpoint works with token + instrument.
        """
        host = "https://api-fxpractice.oanda.com" if self.environment == "practice" else "https://api-fxtrade.oanda.com"
        instrument = f"{base.upper()}_{quote.upper()}"
        url = f"{host}/v3/instruments/{instrument}/candles"

        # Oanda uses RFC3339 timestamps; we request daily candles between start and end.
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {
            "price": "M",             # mid
            "granularity": granularity,
            "from": datetime(start.year, start.month, start.day, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
            "to": (datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
        }

        js = _retry_get_json(url, params=params, headers=headers)

        candles = js.get("candles", [])
        if not candles:
            # Oanda errors can come in 'errorMessage'
            if "errorMessage" in js:
                raise RuntimeError(f"Oanda error: {js['errorMessage']}")
            raise RuntimeError("Oanda returned no candles.")

        rows = []
        for c in candles:
            if not c.get("complete", True):
                continue
            t = c.get("time")
            if not t:
                continue
            obs_d = pd.to_datetime(t, errors="coerce").date()
            if obs_d < start or obs_d > end:
                continue
            mid = c.get("mid", {})
            close = mid.get("c")
            if close is None:
                continue
            rows.append({
                "curve_type": curve_type,
                "base_ccy": base.upper(),
                "quote_ccy": quote.upper(),
                "obs_date": pd.Timestamp(obs_d),
                "spot": float(close),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("Oanda candles parsed empty after filtering.")
        return _ensure_sorted_unique(df)


@dataclass(frozen=True)
class PolygonFxProvider:
    api_key: str

    def fetch_spot_series(self, base: str, quote: str, start: date, end: date, curve_type: str = "Exchange") -> pd.DataFrame:
        """
        Polygon aggregates for FX use ticker like C:USDJPY.
        """
        ticker = f"C:{base.upper()}{quote.upper()}"
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self.api_key}

        js = _retry_get_json(url, params=params)
        if js.get("status") not in {"OK", "DELAYED"}:
            raise RuntimeError(f"Polygon status not OK: {js}")

        results = js.get("results", [])
        if not results:
            raise RuntimeError(f"Polygon returned no results for {ticker}.")

        rows = []
        for r in results:
            # Polygon uses ms epoch in 't'
            ts_ms = r.get("t")
            close = r.get("c")
            if ts_ms is None or close is None:
                continue
            obs_d = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).date()
            rows.append({
                "curve_type": curve_type,
                "base_ccy": base.upper(),
                "quote_ccy": quote.upper(),
                "obs_date": pd.Timestamp(obs_d),
                "spot": float(close),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("Polygon results parsed empty.")
        return _ensure_sorted_unique(df)


# -----------------------
# Test runner
# -----------------------

def main() -> None:
    load_dotenv()

    alpha_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    oanda_key = os.getenv("OANDA_API_KEY")
    polygon_key = os.getenv("POLYGON_API_KEY")

    # Choose a small range to avoid throttling
    start = date(2026, 2, 2)
    end = date(2026, 3, 2)

    base, quote = "USD", "JPY"
    out_dir = Path(__file__).resolve().parent.parent / "data" / "provider_tests"

    providers = {}

    if alpha_key:
        providers["alpha_vantage"] = AlphaVantageFxProvider(alpha_key, outputsize="compact")
    if finnhub_key:
        providers["finnhub"] = FinnhubFxProvider(finnhub_key, venue="OANDA")
    if oanda_key:
        providers["oanda"] = OandaFxProvider(oanda_key, environment="practice")
    if polygon_key:
        providers["polygon"] = PolygonFxProvider(polygon_key)

    if not providers:
        raise RuntimeError("No provider keys found. Put them in .env and try again.")

    for name, p in providers.items():
        print(f"\n=== Testing {name} ===")
        try:
            df = p.fetch_spot_series(base, quote, start, end, curve_type="Exchange")
            print(df.head())
            print(f"rows={len(df)}  from={df['obs_date'].min().date()}  to={df['obs_date'].max().date()}")
            write_fx_spot_headerless_csv(df, out_dir / f"{base}{quote}_{name}.csv")
            print(f"Wrote: {out_dir / f'{base}{quote}_{name}.csv'}")
        except Exception as e:
            print(f"[FAILED] {name}: {e}")

if __name__ == "__main__":
    main()