import pandas as pd
import numpy as np


def build_panel(dc_path, elec_path, water_path, dc_cutoff=2020):
    dc    = pd.read_csv(dc_path)
    elec  = pd.read_csv(elec_path)
    water = pd.read_csv(water_path)

    confirmed = ["reported", "operational", "inferred"]
    dc = dc[dc["year_status"].isin(confirmed)]
    dc = dc[(dc["year"] >= 1990) & (dc["year"] <= dc_cutoff)].dropna(subset=["year"])
    dc["year"] = dc["year"].astype(int)

    openings = dc.groupby(["state_abbrev", "year"]).size().reset_index(name="openings")

    comm = elec[elec["sector"] == "Commercial"][["state_abbrev", "year", "electricity_usage_mwh"]]

    # electricity runs to 2024, extend the panel that far so post-2020 elec growth
    # is available as a signal feature during the forecast rollout
    all_states = comm["state_abbrev"].unique()
    elec_max_year = int(comm["year"].max())

    grid = (
        pd.MultiIndex.from_product([all_states, range(1990, elec_max_year + 1)], names=["state_abbrev", "year"])
        .to_frame(index=False)
    )

    panel = (
        grid
        .merge(openings, on=["state_abbrev", "year"], how="left")
        .merge(comm,     on=["state_abbrev", "year"], how="left")
        .merge(water[["state_abbrev", "year", "water_usage_mgal"]], on=["state_abbrev", "year"], how="left")
    )

    panel["openings"] = panel["openings"].fillna(0).astype(int)
    panel = panel.sort_values(["state_abbrev", "year"]).reset_index(drop=True)

    # cumulative only counts confirmed DCs - post-2020 rows carry the 2020 value forward
    def _cum(g):
        return g["openings"].where(g["year"] <= dc_cutoff, 0).cumsum()

    panel["cum_dcs"] = panel.groupby("state_abbrev", group_keys=False).apply(_cum)

    return panel


def us_totals(panel, cutoff=2020):
    sub = panel[panel["year"] <= cutoff]
    return (
        sub.groupby("year")
        .agg(
            openings=("openings", "sum"),
            cum_dcs=("cum_dcs", "sum"),
            electricity_usage_mwh=("electricity_usage_mwh", "sum"),
            water_usage_mgal=("water_usage_mgal", "sum"),
        )
        .reset_index()
    )
