from pathlib import Path
from fxmd import parse_fx_spot, parse_fx_vol


from datetime import date
from fxmd.providers import AlphaVantageFxProvider, FinnhubFxProvider, write_fx_spot_headerless_csv



DATA_DIR = Path(__file__).resolve().parent.parent / "data"

spot = parse_fx_spot(DATA_DIR / "fx_spot_sim.csv")
meta, vol_long = parse_fx_vol(DATA_DIR / "fx_vol_sim.csv")

print(spot.head())
print(meta.head())
print(vol_long.head())



start = date(2026, 2, 2)
end   = date(2026, 3, 4)

# 1) Alpha Vantage
av = AlphaVantageFxProvider(api_key="YOUR_ALPHA_VANTAGE_KEY", outputsize="full")
df_av = av.fetch_spot_series("USD", "JPY", start, end)
write_fx_spot_headerless_csv(df_av, "data/usdjpy_alpha_vantage.csv")

# 2) Finnhub
fh = FinnhubFxProvider(api_key="YOUR_FINNHUB_KEY", venue="OANDA")
df_fh = fh.fetch_spot_series("USD", "JPY", start, end, resolution="D")
write_fx_spot_headerless_csv(df_fh, "data/usdjpy_finnhub.csv")

print(df_av.head())
print(df_fh.head())