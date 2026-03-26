import pandas as pd
import numpy as np


def add_features(panel, dc_cutoff=2020):
    df = panel.sort_values(["state_abbrev", "year"]).copy()
    g  = df.groupby("state_abbrev")

    for lag in [1, 2, 3]:
        df[f"open_lag{lag}"] = g["openings"].shift(lag)

    df["cum_lag1"] = g["cum_dcs"].shift(1)
    df["cum_lag2"] = g["cum_dcs"].shift(2)

    df["roll3"] = g["openings"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["roll5"] = g["openings"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())

    # exponential smoothed trend on openings - less noisy than raw lags
    df["open_ema3"] = g["openings"].transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean())

    df["elec_lag1"]      = g["electricity_usage_mwh"].shift(1)
    df["elec_growth"]    = g["electricity_usage_mwh"].pct_change()
    # 3yr rolling electricity growth captures sustained acceleration, not single-year noise
    df["elec_growth_3yr"] = g["electricity_usage_mwh"].transform(
        lambda x: x.pct_change().rolling(3, min_periods=1).mean()
    )
    df["water_lag1"] = g["water_usage_mgal"].shift(1)

    df["year_num"] = df["year"] - 2000

    # state rank by total DCs - stable proxy for market size
    state_totals = df[df["year"] <= dc_cutoff].groupby("state_abbrev")["openings"].sum()
    df["state_rank"] = df["state_abbrev"].map(state_totals.rank(ascending=False))

    # national momentum: 3yr rolling average of openings, lagged 1 year
    nat = df[df["year"] <= dc_cutoff].groupby("year")["openings"].sum()
    nat_mom = nat.rolling(3, min_periods=1).mean().shift(1)
    df["national_momentum"] = df["year"].map(nat_mom.to_dict())

    return df


DC_FEATS = [
    "open_lag1", "open_lag2", "open_lag3",
    "cum_lag1", "cum_lag2",
    "roll3", "roll5", "open_ema3",
    "elec_lag1", "elec_growth", "elec_growth_3yr",
    "year_num", "state_rank", "national_momentum",
]

ELEC_FEATS  = ["cum_lag1", "elec_lag1", "elec_growth", "year_num", "state_rank"]
WATER_FEATS = ["cum_lag1", "water_lag1", "year_num", "state_rank"]
