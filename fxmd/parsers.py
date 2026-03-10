# fx_market_data_parsers.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


DATE_FORMATS = ("%d/%m/%Y", "%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y")


# ---------------------------
# Low-level helpers
# ---------------------------

def _norm(s: Any) -> str:
    return "" if s is None else str(s).strip()

def _to_date(s: str) -> pd.Timestamp:
    s = _norm(s)
    if not s:
        return pd.NaT
    for fmt in DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(s, fmt).date())
        except ValueError:
            pass
    return pd.to_datetime(s, errors="coerce")

def _to_float(x: str) -> Optional[float]:
    x = _norm(x)
    if x == "" or x.upper() == "NA":
        return None
    try:
        return float(x)
    except ValueError:
        return None

def _to_int(x: str) -> Optional[int]:
    f = _to_float(x)
    if f is None:
        return None
    if abs(f - round(f)) < 1e-12:
        return int(round(f))
    return None

def _read_rows(path: str | Path) -> List[List[str]]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return [[_norm(c) for c in row] for row in csv.reader(f)]

def _is_template_row(first_cell: str) -> bool:
    return first_cell.strip().lower() in {"field", "description", "type", "constraints", "example"}

def _looks_like_fxspot_header(row: List[str]) -> bool:
    # template headers
    r = {c.lower() for c in row}
    return {"curve type", "base currency id", "observation date", "quote currency id", "exchange rate"}.issubset(r)

def _looks_like_fxvol_header(row: List[str]) -> bool:
    # template headers for the vol sheet
    r = {c.lower() for c in row}
    return {"curve type", "base currency id", "observation date", "quote currency id"}.issubset(r) and (
        "delta" in r or "tte interp type" in r
    )


# ---------------------------
# 1) FX Spot parser
# ---------------------------

def parse_fx_spot(path: str | Path) -> pd.DataFrame:
    """
    Supports both:
      - header template CSV (with 'Curve Type', 'Base Currency ID', ...)
      - headerless CSV rows like:
            Exchange, USD, 02/02/2026, JPY, 155.53
    """
    rows = _read_rows(path)
    rows = [r for r in rows if any(c != "" for c in r)]  # drop empty rows

    if not rows:
        return pd.DataFrame(columns=["curve_type", "base_ccy", "quote_ccy", "obs_date", "spot"])

    # Drop template metadata rows if present
    rows = [r for r in rows if not _is_template_row(r[0] if r else "")]

    # Determine header vs headerless
    if rows and _looks_like_fxspot_header(rows[0]):
        header = rows[0]
        body = rows[1:]
        df = pd.DataFrame(body, columns=header)

        df = df.rename(columns={
            "Curve Type": "curve_type",
            "Base Currency ID": "base_ccy",
            "Quote Currency ID": "quote_ccy",
            "Observation Date": "obs_date",
            "Exchange Rate": "spot",
        })
    else:
        # headerless positional schema: [curve_type, base, obs_date, quote, spot]
        body = rows
        df = pd.DataFrame(body, columns=["curve_type", "base_ccy", "obs_date", "quote_ccy", "spot"])

    # Normalize
    df["obs_date"] = df["obs_date"].astype(str).map(_to_date)
    df["base_ccy"] = df["base_ccy"].astype(str).str.upper().str.strip()
    df["quote_ccy"] = df["quote_ccy"].astype(str).str.upper().str.strip()
    df["spot"] = pd.to_numeric(df["spot"], errors="coerce")

    # Validate
    bad = df[["curve_type", "base_ccy", "quote_ccy", "obs_date", "spot"]].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"FX spot parsing: invalid rows detected. Examples:\n{df.loc[bad].head(10)}")

    return df.reset_index(drop=True)


# ---------------------------
# 2) FX Vol surface parser
# ---------------------------

@dataclass(frozen=True)
class FxVolRowParsed:
    meta: Dict[str, Any]
    expiries: List[int]      # in days (as in your sample)
    vols: List[float]        # implied vols


def _detect_expiry_block(tokens: List[str]) -> Tuple[int, int]:
    """
    Find a contiguous block of expiries (integers, typically increasing) of length >= 5.
    Returns (start_idx, end_idx_exclusive).
    """
    ints = [_to_int(t) for t in tokens]

    best = None  # (start,end)
    n = len(tokens)

    i = 0
    while i < n:
        if ints[i] is None:
            i += 1
            continue

        # start candidate run
        j = i
        run = []
        while j < n and ints[j] is not None:
            run.append(ints[j])
            j += 1

        # run length check and monotonicity check
        if len(run) >= 5:
            # allow non-decreasing (some templates repeat?) but usually strictly increasing
            ok = all(run[k] <= run[k+1] for k in range(len(run)-1))
            # also require that values look like TTEs (positive and not tiny like 0)
            plausible = all(x >= 1 for x in run)
            if ok and plausible:
                # pick the first long plausible run (usually unique)
                best = (i, j)
                break

        i = j + 1

    if best is None:
        raise ValueError(
            "Could not detect expiry block (expected a long run of integer TTEs like 1,7,30,60,...)."
        )
    return best


def _parse_fxvol_row(tokens: List[str]) -> FxVolRowParsed:
    tokens = [t for t in tokens if t != ""]  # drop empty trailing columns

    # detect expiries block
    exp_start, exp_end = _detect_expiry_block(tokens)
    expiries = [int(_to_int(t)) for t in tokens[exp_start:exp_end]]  # safe

    # remaining tokens after expiries should be vols (floats), but can include blanks
    vol_tokens = tokens[exp_end:]
    vols = []
    for t in vol_tokens:
        f = _to_float(t)
        if f is None:
            continue
        vols.append(float(f))

    # We expect at least same count as expiries (often exactly equal)
    if len(vols) < len(expiries):
        raise ValueError(
            f"Detected {len(expiries)} expiries but only {len(vols)} vols. "
            f"Row prefix (first 25 tokens): {tokens[:25]}"
        )

    # If there are extra floats after (unlikely), trim
    vols = vols[:len(expiries)]

    # Meta = everything before expiry block
    meta_tokens = tokens[:exp_start]

    # Extract common fields by pattern/position:
    # In your sample the first four are:
    #   [curve_type_code, base_currency_id-ish, obs_date, quote_currency_id, ...]
    meta: Dict[str, Any] = {}
    if len(meta_tokens) >= 4:
        meta["curve_type"] = meta_tokens[0]
        meta["base_ccy_id"] = meta_tokens[1]
        meta["obs_date"] = _to_date(meta_tokens[2])
        meta["quote_ccy_id"] = meta_tokens[3]

    # Delta: commonly appears as last float in (0,1) before expiries.
    delta = None
    for t in reversed(meta_tokens):
        f = _to_float(t)
        if f is not None and 0.0 < f < 1.0:
            delta = f
            break
    meta["delta"] = delta

    # Keep full meta tokens too (for auditability / later schema hardening)
    meta["meta_tokens"] = meta_tokens

    return FxVolRowParsed(meta=meta, expiries=expiries, vols=vols)


def parse_fx_vol(path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Supports:
      - template-header CSV (rare in your exports)
      - headerless wide rows like in your screenshots.

    Returns:
      meta_df: per-row meta (incl. raw meta_tokens)
      quotes_df: long format: one row per (obs_date, pair, delta, expiry_days, vol)
    """
    rows = _read_rows(path)
    rows = [r for r in rows if any(c != "" for c in r)]  # drop empty rows
    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    # drop template metadata rows
    rows = [r for r in rows if not _is_template_row(r[0] if r else "")]

    # If header is present, we could parse via pandas columns;
    # but your real samples are headerless, so we focus on robust headerless parsing.
    if rows and _looks_like_fxvol_header(rows[0]):
        # If you ever get a true header vol file, you can plug in a strict schema here.
        # For now we fall back to headerless parsing to be safe.
        rows = rows[1:]

    parsed: List[FxVolRowParsed] = []
    for idx, r in enumerate(rows):
        try:
            parsed.append(_parse_fxvol_row(r))
        except Exception as e:
            raise ValueError(f"FX vol parsing failed at row {idx+1}. First 30 cells: {r[:30]}\n{e}") from e

    # meta_df
    meta_df = pd.DataFrame([p.meta for p in parsed])
    # normalize ccy IDs
    for c in ["base_ccy_id", "quote_ccy_id"]:
        if c in meta_df.columns:
            meta_df[c] = meta_df[c].astype(str).str.upper().str.strip()

    # quotes_df long
    long_rows = []
    for p in parsed:
        curve_type = p.meta.get("curve_type")
        base = p.meta.get("base_ccy_id")
        quote = p.meta.get("quote_ccy_id")
        obs_date = p.meta.get("obs_date")
        delta = p.meta.get("delta")

        for tte, vol in zip(p.expiries, p.vols):
            long_rows.append({
                "curve_type": curve_type,
                "base_ccy_id": base,
                "quote_ccy_id": quote,
                "obs_date": obs_date,
                "delta": delta,
                "expiry_days": int(tte),
                "vol": float(vol),
            })

    quotes_df = pd.DataFrame(long_rows)

    # sanity checks
    bad = quotes_df[["curve_type", "base_ccy_id", "quote_ccy_id", "obs_date", "expiry_days", "vol"]].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"FX vol parsing produced invalid long rows. Examples:\n{quotes_df.loc[bad].head(10)}")

    return meta_df.reset_index(drop=True), quotes_df.reset_index(drop=True)