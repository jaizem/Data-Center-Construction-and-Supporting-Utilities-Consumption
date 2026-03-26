import numpy as np
import pandas as pd
from src.features import DC_FEATS, ELEC_FEATS, WATER_FEATS


def rollout(models, panel, forecast_years=range(2021, 2031), dc_cutoff=2020):
    dc_model    = models["dc"]
    elec_model  = models["elec"]
    water_model = models["water"]

    nat_by_year = panel[panel["year"] <= dc_cutoff].groupby("year")["openings"].sum().to_dict()
    state_ranks = (
        panel[panel["year"] <= dc_cutoff]
        .groupby("state_abbrev")["openings"].sum()
        .rank(ascending=False).to_dict()
    )

    # actual electricity values 2021-2024 -- we have real data here and use it
    # as a feature signal rather than asking the model to predict something we know
    elec_actuals = (
        panel[panel["year"] > dc_cutoff]
        .set_index(["state_abbrev", "year"])["electricity_usage_mwh"]
        .to_dict()
    )

    rows = []

    for state in panel["state_abbrev"].unique():
        hist = panel[(panel["state_abbrev"] == state) & (panel["year"] <= dc_cutoff)].sort_values("year")
        if hist.empty:
            continue

        carry       = hist.tail(5).copy()
        elec_carry  = hist["electricity_usage_mwh"].dropna().iloc[-1] if hist["electricity_usage_mwh"].notna().any() else np.nan
        water_carry = hist["water_usage_mgal"].dropna().iloc[-1]       if hist["water_usage_mgal"].notna().any()      else np.nan
        rank        = state_ranks.get(state, 25)

        for yr in forecast_years:
            prev_elec = elec_carry

            actual_elec = elec_actuals.get((state, yr))
            if actual_elec is not None and not np.isnan(float(actual_elec)):
                # use real electricity data where we have it (2021-2024)
                elec_carry  = float(actual_elec)
                elec_growth = (elec_carry / prev_elec - 1) if (prev_elec and prev_elec > 0) else 0.0
            else:
                # after 2024 the electricity model predicts next year's value
                # we use that prediction as the feature for the DC model in the following year
                X_elec_forecast = np.array([[
                    carry.iloc[-1]["cum_dcs"],
                    elec_carry,
                    elec_growth if "elec_growth" in dir() else 0.0,
                    yr - 2000,
                    rank,
                ]])
                pred_elec_next = float(elec_model.predict(X_elec_forecast)[0])
                elec_growth = (pred_elec_next / elec_carry - 1) if elec_carry > 0 else 0.0
                elec_carry  = pred_elec_next

            # 3-year rolling electricity growth for the longer-term trend feature
            recent = [elec_actuals.get((state, yr - i)) for i in [1, 2, 3]]
            recent = [float(v) for v in recent if v is not None and not np.isnan(float(v))]
            if len(recent) >= 2:
                elec_growth_3yr = np.mean(np.diff(recent) / np.array(recent[:-1]))
            else:
                elec_growth_3yr = elec_growth

            X_dc = np.array([[
                carry.iloc[-1]["openings"],
                carry.iloc[-2]["openings"] if len(carry) >= 2 else 0,
                carry.iloc[-3]["openings"] if len(carry) >= 3 else 0,
                carry.iloc[-1]["cum_dcs"],
                carry.iloc[-2]["cum_dcs"]  if len(carry) >= 2 else 0,
                carry["openings"].tail(3).mean(),
                carry["openings"].tail(5).mean(),
                carry["openings"].ewm(span=3, adjust=False).mean().iloc[-1],
                elec_carry,
                elec_growth,
                elec_growth_3yr,
                yr - 2000,
                rank,
                np.mean([nat_by_year.get(yr - i, 0) for i in [1, 2, 3]]),
            ]])

            pred_open = max(0.0, round(float(dc_model.predict(X_dc)[0])))
            new_cum   = carry.iloc[-1]["cum_dcs"] + pred_open
            nat_by_year[yr] = nat_by_year.get(yr, 0) + pred_open

            # electricity and water outputs for reporting -- both from their ML models
            X_elec  = np.array([[new_cum, elec_carry, elec_growth, yr - 2000, rank]])
            X_water = np.array([[new_cum, water_carry, yr - 2000, rank]])

            pred_elec  = float(elec_model.predict(X_elec)[0])  if not np.isnan(elec_carry)  else np.nan
            pred_water = float(water_model.predict(X_water)[0]) if not np.isnan(water_carry) else np.nan

            # feed water model output back so next year's water prediction uses
            # a realistic starting point rather than the 2020 value forever
            if not np.isnan(pred_water):
                water_carry = pred_water

            rows.append({
                "state_abbrev":          state,
                "year":                  yr,
                "openings":              pred_open,
                "cum_dcs":               new_cum,
                "electricity_usage_mwh": pred_elec,
                "water_usage_mgal":      pred_water,
            })

            new_row = pd.DataFrame([{
                "state_abbrev": state, "year": yr,
                "openings": pred_open, "cum_dcs": new_cum,
                "electricity_usage_mwh": pred_elec,
                "water_usage_mgal": pred_water,
                "state_rank": rank,
            }])
            carry = pd.concat([carry, new_row], ignore_index=True)

    return pd.DataFrame(rows)


def us_forecast(state_fc):
    return (
        state_fc.groupby("year")
        .agg(
            openings=("openings", "sum"),
            cum_dcs=("cum_dcs", "sum"),
            electricity_usage_mwh=("electricity_usage_mwh", lambda x: x.dropna().sum()),
            water_usage_mgal=("water_usage_mgal", lambda x: x.dropna().sum()),
        )
        .reset_index()
    )
