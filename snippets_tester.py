from fx_market_data_parsers import parse_fx_spot, parse_fx_vol

spot = parse_fx_spot("fx_spot_sim.csv")
meta, vol_long = parse_fx_vol("fx_vol_sim.csv")

print(spot.head())
print(meta.head())
print(vol_long.head())


