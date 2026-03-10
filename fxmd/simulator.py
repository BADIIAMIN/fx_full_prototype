# fx_market_data_simulator.py
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd


# -----------------------------
# Utilities
# -----------------------------

def _fmt_date(d: date, fmt: str = "%d/%m/%Y") -> str:
    return d.strftime(fmt)

def _is_business_day(d: date) -> bool:
    # simple: Mon-Fri. (You can plug holiday calendars later)
    return d.weekday() < 5

def _business_days(start: date, n: int) -> List[date]:
    out = []
    d = start
    while len(out) < n:
        if _is_business_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _normal(rng: random.Random, mu: float = 0.0, sigma: float = 1.0) -> float:
    # Box-Muller using random.gauss is fine too, but keep explicit
    return rng.gauss(mu, sigma)


# -----------------------------
# 1) FX Spot simulator
# -----------------------------

@dataclass(frozen=True)
class FxSpotSimConfig:
    curve_type: str = "Exchange"
    base_ccy: str = "USD"
    quote_ccy: str = "JPY"
    start_date: date = date(2026, 2, 2)
    n_obs: int = 30                  # number of business days
    s0: float = 155.53               # initial spot
    mu: float = 0.00                 # drift (annualised)
    sigma: float = 0.10              # vol (annualised)
    day_count: int = 252             # business-day year
    date_format: str = "%d/%m/%Y"
    seed: int = 42


def simulate_fx_spot_timeseries(cfg: FxSpotSimConfig) -> pd.DataFrame:
    rng = random.Random(cfg.seed)
    dates = _business_days(cfg.start_date, cfg.n_obs)

    s = cfg.s0
    rows = []
    dt = 1.0 / cfg.day_count

    for d in dates:
        # GBM in discrete time
        z = _normal(rng)
        s = s * math.exp((cfg.mu - 0.5 * cfg.sigma**2) * dt + cfg.sigma * math.sqrt(dt) * z)

        rows.append({
            "curve_type": cfg.curve_type,
            "base_ccy": cfg.base_ccy,
            "obs_date": d,
            "quote_ccy": cfg.quote_ccy,
            "spot": s
        })

    df = pd.DataFrame(rows)
    return df


def write_fx_spot_csv_headerless(df: pd.DataFrame, path: str | Path, date_format: str = "%d/%m/%Y") -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for _, r in df.iterrows():
            w.writerow([
                r["curve_type"],
                r["base_ccy"],
                _fmt_date(r["obs_date"], date_format),
                r["quote_ccy"],
                f"{float(r['spot']):.2f}",
            ])


# -----------------------------
# 2) FX Vol surface simulator
# -----------------------------

@dataclass(frozen=True)
class FxVolSimConfig:
    # Identifiers similar to screenshot
    curve_type: str = "FXDeltaVolMtx"
    surface_id: str = "USD-CALL"     # or USD-PUT, USD, etc.
    base_ccy: str = "USD"
    quote_ccy: str = "JPY"
    start_date: date = date(2026, 2, 2)
    n_obs: int = 10
    date_format: str = "%d/%m/%Y"
    seed: int = 7

    # Meta tokens (kept as strings to match the export “wide row” feel)
    tte_interp: str = "cubic"
    tte_extrap: str = "near"
    delta_interp: str = "lin"
    delta_extrap: str = "near"
    cp: str = "CALL"                # CALL/PUT/ATMF etc in your file variants
    delta_def: str = "FORWARD"      # FORWARD
    premium_adj: str = "INCLUDED"   # INCLUDED/EXCLUDED
    days_per_annum: int = 360
    holiday_set: str = "NONE"
    tte_units: str = "DAYS"
    bus_day_conv: str = "NA"
    eom_rule: str = "PREV"
    some_flag: str = "FALSE"        # matches the boolean-ish token near the end

    # Smile grid
    delta: float = 0.25             # 0.1 / 0.25 / 0.5 ... (your file shows 0.1 and 0.25)
    expiries_days: Tuple[int, ...] = (1, 7, 30, 60, 90, 180, 270, 365, 730, 1095, 1825, 2555, 3650, 4380, 5475, 7300, 9125, 10950)

    # Vol model parameters (simple but realistic shape)
    base_atm_vol: float = 0.10      # around 10% like your screenshot
    term_slope: float = -0.015      # longer tenor slightly lower/higher
    term_curv: float = 0.010        # curvature in ln(T)
    smile_amp: float = 0.020        # “delta” dependence amplitude
    obs_noise: float = 0.003        # day-to-day noise (absolute vol)
    min_vol: float = 0.01
    max_vol: float = 1.50


def _term_structure_vol(T_years: float, base: float, slope: float, curv: float) -> float:
    # smooth term structure in log-maturity
    x = math.log(max(T_years, 1e-6))
    return base + slope * x + curv * x * x

def _smile_adjust(delta: float, amp: float) -> float:
    # simple symmetric smile around 0.5 (ATM)
    # delta in (0,1); smile higher away from 0.5
    return amp * (abs(delta - 0.5) / 0.5) ** 1.2

def simulate_fx_vol_rows(cfg: FxVolSimConfig) -> pd.DataFrame:
    rng = random.Random(cfg.seed)
    dates = _business_days(cfg.start_date, cfg.n_obs)

    dt = 1.0 / 252.0
    atm_level = cfg.base_atm_vol

    rows = []
    for d in dates:
        # evolve atm level slowly (mean-reverting-ish random walk)
        atm_level = _clamp(atm_level + _normal(rng, 0.0, cfg.obs_noise) * math.sqrt(dt), 0.02, 0.60)

        vols = []
        for tte in cfg.expiries_days:
            T = max(tte / 365.0, 1.0 / 365.0)
            v = _term_structure_vol(T, atm_level, cfg.term_slope, cfg.term_curv)
            v += _smile_adjust(cfg.delta, cfg.smile_amp)
            v += _normal(rng, 0.0, cfg.obs_noise)
            v = _clamp(v, cfg.min_vol, cfg.max_vol)
            vols.append(v)

        rows.append({
            "curve_type": cfg.curve_type,
            "surface_id": cfg.surface_id,
            "obs_date": d,
            "quote_ccy": cfg.quote_ccy,
            "tte_interp": cfg.tte_interp,
            "tte_extrap": cfg.tte_extrap,
            "delta_interp": cfg.delta_interp,
            "delta_extrap": cfg.delta_extrap,
            "cp": cfg.cp,
            "delta_def": cfg.delta_def,
            "premium_adj": cfg.premium_adj,
            "days_per_annum": cfg.days_per_annum,
            "holiday_set": cfg.holiday_set,
            "tte_units": cfg.tte_units,
            "bus_day_conv": cfg.bus_day_conv,
            "eom_rule": cfg.eom_rule,
            "some_flag": cfg.some_flag,
            "delta": cfg.delta,
            "expiries_days": list(cfg.expiries_days),
            "vols": vols,
        })

    return pd.DataFrame(rows)


def write_fx_vol_csv_headerless_wide(df: pd.DataFrame, path: str | Path, date_format: str = "%d/%m/%Y") -> None:
    """
    Writes *one row per surface snapshot* in the same wide style:
        [meta tokens ...] + [expiries days ...] + [vol floats ...]
    """
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        for _, r in df.iterrows():
            meta = [
                r["curve_type"],           # e.g. FXDeltaVolMtx
                r["surface_id"],           # e.g. USD-CALL
                _fmt_date(r["obs_date"], date_format),
                r["quote_ccy"],            # JPY
                r["tte_interp"],
                r["tte_extrap"],
                r["delta_interp"],
                r["delta_extrap"],
                r["cp"],
                r["delta_def"],
                r["premium_adj"],
                str(int(r["days_per_annum"])),
                r["holiday_set"],
                r["tte_units"],
                r["bus_day_conv"],
                r["eom_rule"],
                r["some_flag"],
                f"{float(r['delta']):.2f}",
            ]
            expiries = [str(int(x)) for x in r["expiries_days"]]
            vols = [f"{float(v):.6f}" for v in r["vols"]]

            w.writerow(meta + expiries + vols)


# -----------------------------
# Demo / CLI-like usage
# -----------------------------

if __name__ == "__main__":
    out_dir = Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) FX Spot
    spot_cfg = FxSpotSimConfig(
        base_ccy="USD",
        quote_ccy="JPY",
        start_date=date(2026, 2, 2),
        n_obs=25,
        s0=155.53,
        mu=0.00,
        sigma=0.12,
        seed=123,
    )
    spot_df = simulate_fx_spot_timeseries(spot_cfg)
    write_fx_spot_csv_headerless(spot_df, out_dir / "fx_spot_sim.csv", date_format=spot_cfg.date_format)

    # 2) FX Vol (example: delta=0.25 call surface)
    vol_cfg = FxVolSimConfig(
        surface_id="USD-CALL",
        base_ccy="USD",
        quote_ccy="JPY",
        start_date=date(2026, 2, 2),
        n_obs=8,
        delta=0.25,
        cp="CALL",
        seed=777,
        base_atm_vol=0.10,
    )
    vol_df = simulate_fx_vol_rows(vol_cfg)
    write_fx_vol_csv_headerless_wide(vol_df, out_dir / "fx_vol_sim.csv", date_format=vol_cfg.date_format)

    print("Wrote:")
    print(" - fx_spot_sim.csv")
    print(" - fx_vol_sim.csv")