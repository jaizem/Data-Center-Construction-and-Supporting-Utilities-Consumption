import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

NAT_DIR   = Path("viz/eda/national")
STATE_DIR = Path("viz/eda/state")

BLUE    = "#1D4ED8"
ORANGE  = "#EA580C"
GREEN   = "#15803D"
GREY    = "#9CA3AF"
RED     = "#DC2626"
L_BLUE  = "#DBEAFE"
L_GREEN = "#DCFCE7"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130,
})

# key moments in the data center / cloud / AI timeline
# these get added to charts as vertical reference lines
MILESTONES = [
    (1993, "Mosaic browser",        "top"),
    (1999, "dot-com peak",          "top"),
    (2004, "Gmail launch",          "bottom"),
    (2006, "AWS launches",          "top"),
    (2008, "iPhone App Store",      "bottom"),
    (2012, "Big Data boom",         "top"),
    (2015, "Cloud mainstream",      "bottom"),
    (2017, "GPU cloud training",    "top"),
    (2020, "COVID remote surge",    "bottom"),
]


def add_milestones(ax, y_range, which="all", skip=None):
    # draws vertical lines with small text labels at the top or bottom
    # y_range is (ymin, ymax) so we can position text correctly
    skip = skip or []
    ymin, ymax = y_range
    span = ymax - ymin

    for year, label, position in MILESTONES:
        if label in skip:
            continue
        xmin, xmax = ax.get_xlim()
        if not (xmin <= year <= xmax):
            continue
        ax.axvline(year, color=GREY, lw=0.8, ls="--", alpha=0.5, zorder=1)
        y = ymax - span * 0.04 if position == "top" else ymin + span * 0.02
        ax.text(year + 0.2, y, label, fontsize=7, color=GREY,
                rotation=90, va="top" if position == "top" else "bottom",
                alpha=0.8, zorder=2)


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


# national charts

def national_dc_growth(panel):
    annual = panel[panel["year"] <= 2020].groupby("year")["openings"].sum().reset_index()
    annual["cum"] = annual["openings"].cumsum()
    annual["ma3"] = annual["openings"].rolling(3, min_periods=1).mean()
    annual["ma5"] = annual["openings"].rolling(5, min_periods=2).mean()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    colors = [BLUE if y >= 2006 else GREY for y in annual["year"]]
    ax.bar(annual["year"], annual["openings"], color=colors, width=0.8, alpha=0.7, label="Annual openings")
    ax.plot(annual["year"], annual["ma3"], color=ORANGE, lw=2, label="3yr moving avg")
    ax.plot(annual["year"], annual["ma5"], color=RED, lw=2, ls="--", label="5yr moving avg")
    add_milestones(ax, (0, annual["openings"].max() * 1.1))
    ax.set_title("Annual U.S. Data Center Openings", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("New Openings")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.fill_between(annual["year"], annual["cum"], color=L_BLUE, alpha=0.9)
    ax.plot(annual["year"], annual["cum"], color=BLUE, lw=2.5)
    add_milestones(ax, (0, annual["cum"].max() * 1.1))
    ax.set_title("Cumulative U.S. Data Centers", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Facilities")
    ax.text(2017, annual["cum"].iloc[-1] * 0.9, f"1,644 by 2020",
            fontsize=9, color=GREY)

    fig.suptitle("U.S. Data Center Build-Out 1990 to 2020", fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, NAT_DIR, "dc_growth")


def national_electricity(us):
    df = us[us["year"] <= 2020].dropna(subset=["electricity_usage_mwh"]).copy()
    df["ma3"] = df["electricity_usage_mwh"].rolling(3, min_periods=1).mean()
    df["yoy"] = df["electricity_usage_mwh"].pct_change() * 100

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    ax.fill_between(df["year"], df["electricity_usage_mwh"] / 1e9, color=L_BLUE, alpha=0.9)
    ax.plot(df["year"], df["electricity_usage_mwh"] / 1e9, color=BLUE, lw=2.5, label="Annual")
    ax.plot(df["year"], df["ma3"] / 1e9, color=ORANGE, lw=2, ls="--", label="3yr moving avg")
    ax.axhline(df["electricity_usage_mwh"].median() / 1e9, color=GREY, lw=1, ls=":",
               label=f"Median: {df['electricity_usage_mwh'].median()/1e9:.2f} B MWh")
    add_milestones(ax, (df["electricity_usage_mwh"].min()/1e9, df["electricity_usage_mwh"].max()/1e9 * 1.05))
    ax.set_title("U.S. Commercial Electricity Demand", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Billion MWh")
    ax.legend(fontsize=9)

    ax = axes[1]
    colors = [GREEN if v >= 0 else RED for v in df["yoy"].fillna(0)]
    ax.bar(df["year"], df["yoy"].fillna(0), color=colors, width=0.8, alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(df["yoy"].mean(), color=ORANGE, lw=1.5, ls="--",
               label=f"Avg YoY: {df['yoy'].mean():.1f}%")
    add_milestones(ax, (df["yoy"].min() - 1, df["yoy"].max() * 1.1))
    ax.set_title("Year-over-Year Change (%)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("% Change")
    ax.legend(fontsize=9)

    fig.suptitle("U.S. Commercial Electricity 1990 to 2020", fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, NAT_DIR, "electricity")


def national_water(us):
    df = us[us["year"] <= 2020].dropna(subset=["water_usage_mgal"]).copy()
    df["ma3"] = df["water_usage_mgal"].rolling(3, min_periods=1).mean()
    df["yoy"] = df["water_usage_mgal"].pct_change() * 100

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    ax.fill_between(df["year"], df["water_usage_mgal"] / 1e3, color=L_GREEN, alpha=0.9)
    ax.plot(df["year"], df["water_usage_mgal"] / 1e3, color=GREEN, lw=2.5, label="Annual")
    ax.plot(df["year"], df["ma3"] / 1e3, color=ORANGE, lw=2, ls="--", label="3yr moving avg")
    ax.axhline(df["water_usage_mgal"].median() / 1e3, color=GREY, lw=1, ls=":",
               label=f"Median: {df['water_usage_mgal'].median()/1e3:.0f} B gal")
    water_milestones = [(y, l, p) for y, l, p in MILESTONES if y >= 2000]
    yspan = (df["water_usage_mgal"].min()/1e3, df["water_usage_mgal"].max()/1e3 * 1.05)
    for year, label, position in water_milestones:
        xmin, xmax = ax.get_xlim()
        ax.axvline(year, color=GREY, lw=0.8, ls="--", alpha=0.5, zorder=1)
        y = yspan[1] * 0.96 if position == "top" else yspan[0] + (yspan[1]-yspan[0])*0.02
        ax.text(year + 0.2, y, label, fontsize=7, color=GREY, rotation=90, va="top", alpha=0.8)
    ax.set_title("U.S. Water Usage 2000 to 2020", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Billion Gallons")
    ax.legend(fontsize=9)

    ax = axes[1]
    colors = [GREEN if v >= 0 else RED for v in df["yoy"].fillna(0)]
    ax.bar(df["year"], df["yoy"].fillna(0), color=colors, width=0.8, alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(df["yoy"].mean(), color=ORANGE, lw=1.5, ls="--",
               label=f"Avg YoY: {df['yoy'].mean():.1f}%")
    ax.set_title("Year-over-Year Change (%)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("% Change")
    ax.legend(fontsize=9)

    fig.suptitle("U.S. Water Usage 2000 to 2020", fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, NAT_DIR, "water")


def national_holistic(us, panel):
    df = us[(us["year"] >= 2000) & (us["year"] <= 2020)].copy()
    df = df.dropna(subset=["electricity_usage_mwh", "water_usage_mgal"])

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(df["year"], df["openings"], color=BLUE, alpha=0.6, width=0.8, label="Openings")
    ma = df["openings"].rolling(3, min_periods=1).mean()
    ax1.plot(df["year"], ma, color=ORANGE, lw=2, label="3yr avg")
    ax1.axhline(df["openings"].median(), color=GREY, lw=1, ls="--",
                label=f"Median: {df['openings'].median():.0f}")
    add_milestones(ax1, (0, df["openings"].max() * 1.15))
    ax1.set_title("Annual DC Openings", fontweight="bold")
    ax1.set_ylabel("New Openings")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(df["year"], df["cum_dcs"], color=L_BLUE, alpha=0.9)
    ax2.plot(df["year"], df["cum_dcs"], color=BLUE, lw=2.5)
    add_milestones(ax2, (0, df["cum_dcs"].max() * 1.15))
    ax2.set_title("Cumulative Data Centers", fontweight="bold")
    ax2.set_ylabel("Total Facilities")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.fill_between(df["year"], df["electricity_usage_mwh"] / 1e9, color=L_BLUE, alpha=0.7)
    ax3.plot(df["year"], df["electricity_usage_mwh"] / 1e9, color=BLUE, lw=2.5, label="Electricity")
    ax3.axhline(df["electricity_usage_mwh"].median() / 1e9, color=GREY, lw=1, ls="--",
                label=f"Median: {df['electricity_usage_mwh'].median()/1e9:.2f} B MWh")
    ax3b = ax3.twinx()
    ax3b.plot(df["year"], df["cum_dcs"], color=ORANGE, lw=1.5, ls=":", alpha=0.7, label="Cum DCs")
    ax3b.set_ylabel("Cumulative DCs", color=ORANGE, fontsize=9)
    ax3b.tick_params(axis="y", labelcolor=ORANGE, labelsize=8)
    ax3.set_title("Electricity + Cumulative DCs", fontweight="bold")
    ax3.set_ylabel("Billion MWh")
    ax3.legend(fontsize=8, loc="upper left")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.fill_between(df["year"], df["water_usage_mgal"] / 1e3, color=L_GREEN, alpha=0.7)
    ax4.plot(df["year"], df["water_usage_mgal"] / 1e3, color=GREEN, lw=2.5, label="Water")
    ax4.axhline(df["water_usage_mgal"].median() / 1e3, color=GREY, lw=1, ls="--",
                label=f"Median: {df['water_usage_mgal'].median()/1e3:.0f} B gal")
    ax4b = ax4.twinx()
    ax4b.plot(df["year"], df["cum_dcs"], color=ORANGE, lw=1.5, ls=":", alpha=0.7, label="Cum DCs")
    ax4b.set_ylabel("Cumulative DCs", color=ORANGE, fontsize=9)
    ax4b.tick_params(axis="y", labelcolor=ORANGE, labelsize=8)
    ax4.set_title("Water Usage + Cumulative DCs", fontweight="bold")
    ax4.set_ylabel("Billion Gallons")
    ax4.legend(fontsize=8, loc="upper left")

    fig.suptitle("National Overview: Data Centers, Electricity and Water 2000 to 2020",
                 fontweight="bold", fontsize=13, y=1.01)
    save(fig, NAT_DIR, "holistic_overview")


def national_correlations(us):
    df = us[(us["year"] >= 2000) & (us["year"] <= 2020)].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(df["cum_dcs"], df["electricity_usage_mwh"] / 1e9,
               s=60, color=BLUE, alpha=0.7, zorder=3)
    z = np.polyfit(df["cum_dcs"], df["electricity_usage_mwh"] / 1e9, 1)
    x_r = np.linspace(df["cum_dcs"].min(), df["cum_dcs"].max(), 100)
    ax.plot(x_r, np.poly1d(z)(x_r), "--", color=ORANGE, lw=2)
    for yr in [2005, 2010, 2015, 2020]:
        row = df[df["year"] == yr]
        if not row.empty:
            ax.annotate(str(yr),
                        xy=(row["cum_dcs"].values[0], row["electricity_usage_mwh"].values[0] / 1e9),
                        xytext=(5, 5), textcoords="offset points", fontsize=8, color=GREY)
    r = df["cum_dcs"].corr(df["electricity_usage_mwh"])
    ax.set_title(f"Cumulative DCs vs Electricity  (r={r:.2f})", fontweight="bold")
    ax.set_xlabel("Cumulative U.S. Data Centers")
    ax.set_ylabel("Commercial Electricity (B MWh)")

    ax = axes[1]
    ax.scatter(df["cum_dcs"], df["water_usage_mgal"] / 1e3,
               s=60, color=GREEN, alpha=0.7, zorder=3)
    z2 = np.polyfit(df["cum_dcs"], df["water_usage_mgal"] / 1e3, 1)
    ax.plot(x_r, np.poly1d(z2)(x_r), "--", color=ORANGE, lw=2)
    for yr in [2005, 2010, 2015, 2020]:
        row = df[df["year"] == yr]
        if not row.empty:
            ax.annotate(str(yr),
                        xy=(row["cum_dcs"].values[0], row["water_usage_mgal"].values[0] / 1e3),
                        xytext=(5, 5), textcoords="offset points", fontsize=8, color=GREY)
    r2 = df["cum_dcs"].corr(df["water_usage_mgal"])
    ax.set_title(f"Cumulative DCs vs Water  (r={r2:.2f})", fontweight="bold")
    ax.set_xlabel("Cumulative U.S. Data Centers")
    ax.set_ylabel("Water Usage (B gallons)")

    fig.suptitle("How Data Center Growth Drives Resource Demand", fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, NAT_DIR, "correlations")


# state level charts

def state_rankings(panel):
    totals = (
        panel[panel["year"] <= 2020]
        .groupby("state_abbrev")["openings"].sum()
        .sort_values(ascending=True).tail(15)
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [ORANGE if i >= 12 else BLUE for i in range(len(totals))]
    bars = ax.barh(totals.index, totals.values, color=colors, alpha=0.85)
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())), va="center", fontsize=9)
    ax.axvline(totals.median(), color=GREY, lw=1.5, ls="--",
               label=f"Median: {totals.median():.0f}")
    ax.set_title("Top 15 States by Data Centers Opened 1990 to 2020", fontweight="bold")
    ax.set_xlabel("Total Openings")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, STATE_DIR, "rankings")


def state_growth_curves(panel, states=None):
    if states is None:
        states = (
            panel[panel["year"] <= 2020]
            .groupby("state_abbrev")["openings"].sum()
            .sort_values(ascending=False).head(8).index.tolist()
        )
    df = panel[panel["year"] <= 2020]

    fig, ax = plt.subplots(figsize=(14, 6))
    palette = plt.cm.tab10(np.linspace(0, 0.9, len(states)))
    for state, color in zip(states, palette):
        s = df[df["state_abbrev"] == state].sort_values("year")
        ax.plot(s["year"], s["cum_dcs"], lw=2.5, color=color, label=state)
        if len(s):
            ax.scatter(s["year"].iloc[-1], s["cum_dcs"].iloc[-1], color=color, s=60, zorder=5)
    add_milestones(ax, (0, df.groupby("state_abbrev")["cum_dcs"].max().max() * 1.1))
    ax.set_title("Cumulative Data Centers by State", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Facilities")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, STATE_DIR, "growth_curves")


def state_holistic(panel, states=None):
    if states is None:
        states = (
            panel[panel["year"] <= 2020]
            .groupby("state_abbrev")["openings"].sum()
            .sort_values(ascending=False).head(6).index.tolist()
        )

    for state in states:
        # cap at 2020 for all three panels
        s = panel[
            (panel["state_abbrev"] == state) &
            (panel["year"] >= 2000) &
            (panel["year"] <= 2020)
        ].sort_values("year")

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        ax = axes[0]
        ax.bar(s["year"], s["openings"], color=BLUE, alpha=0.7, width=0.8, label="Annual openings")
        ma = s["openings"].rolling(3, min_periods=1).mean()
        ax.plot(s["year"], ma, color=ORANGE, lw=2, label="3yr avg")
        ax.axhline(s["openings"].median(), color=GREY, lw=1, ls="--",
                   label=f"Median: {s['openings'].median():.1f}")
        add_milestones(ax, (0, s["openings"].max() * 1.15))
        ax.set_ylabel("DC Openings")
        ax.set_title(f"{state} -- Data Centers, Electricity and Water 2000 to 2020", fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")

        s_e = s.dropna(subset=["electricity_usage_mwh"])
        ax = axes[1]
        ax.fill_between(s_e["year"], s_e["electricity_usage_mwh"] / 1e6, color=L_BLUE, alpha=0.9)
        ax.plot(s_e["year"], s_e["electricity_usage_mwh"] / 1e6, color=BLUE, lw=2)
        ma_e = (s_e["electricity_usage_mwh"] / 1e6).rolling(3, min_periods=1).mean()
        ax.plot(s_e["year"], ma_e, color=ORANGE, lw=1.5, ls="--", label="3yr avg")
        ax.axhline(s_e["electricity_usage_mwh"].median() / 1e6, color=GREY, lw=1, ls=":",
                   label=f"Median: {s_e['electricity_usage_mwh'].median()/1e6:.1f}M MWh")
        ax.set_ylabel("Commercial Electricity (M MWh)")
        ax.legend(fontsize=8, loc="upper left")

        s_w = s.dropna(subset=["water_usage_mgal"])
        ax = axes[2]
        if not s_w.empty:
            ax.fill_between(s_w["year"], s_w["water_usage_mgal"] / 1e3, color=L_GREEN, alpha=0.9)
            ax.plot(s_w["year"], s_w["water_usage_mgal"] / 1e3, color=GREEN, lw=2)
            ma_w = (s_w["water_usage_mgal"] / 1e3).rolling(3, min_periods=1).mean()
            ax.plot(s_w["year"], ma_w, color=ORANGE, lw=1.5, ls="--", label="3yr avg")
            ax.axhline(s_w["water_usage_mgal"].median() / 1e3, color=GREY, lw=1, ls=":",
                       label=f"Median: {s_w['water_usage_mgal'].median()/1e3:.1f} B gal")
            ax.legend(fontsize=8, loc="upper left")
        ax.set_ylabel("Water Usage (B gallons)")
        ax.set_xlabel("Year")

        fig.tight_layout()
        save(fig, STATE_DIR, f"holistic_{state.lower()}")


def state_resource_scatter(panel):
    df = panel[
        (panel["year"] >= 2015) & (panel["year"] <= 2020)
    ].dropna(subset=["electricity_usage_mwh"])

    by_state = df.groupby("state_abbrev").agg(
        avg_cum_dcs=("cum_dcs", "mean"),
        avg_elec=("electricity_usage_mwh", "mean"),
        avg_water=("water_usage_mgal", "mean"),
    ).reset_index()

    top_states = by_state.nlargest(10, "avg_cum_dcs")["state_abbrev"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(by_state["avg_cum_dcs"], by_state["avg_elec"] / 1e6,
               s=50, color=BLUE, alpha=0.5, zorder=2)
    for _, row in by_state[by_state["state_abbrev"].isin(top_states)].iterrows():
        ax.annotate(row["state_abbrev"],
                    xy=(row["avg_cum_dcs"], row["avg_elec"] / 1e6),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Avg Cumulative DCs vs Avg Electricity 2015 to 2020", fontweight="bold")
    ax.set_xlabel("Avg Cumulative Data Centers")
    ax.set_ylabel("Avg Commercial Electricity (M MWh)")

    by_w = by_state.dropna(subset=["avg_water"])
    ax = axes[1]
    ax.scatter(by_w["avg_cum_dcs"], by_w["avg_water"] / 1e3,
               s=50, color=GREEN, alpha=0.5, zorder=2)
    for _, row in by_w[by_w["state_abbrev"].isin(top_states)].iterrows():
        ax.annotate(row["state_abbrev"],
                    xy=(row["avg_cum_dcs"], row["avg_water"] / 1e3),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Avg Cumulative DCs vs Avg Water 2015 to 2020", fontweight="bold")
    ax.set_xlabel("Avg Cumulative Data Centers")
    ax.set_ylabel("Avg Water (B gallons)")

    fig.suptitle("State Level Resource Demand vs Data Center Presence", fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, STATE_DIR, "resource_scatter")


def run_all_eda(panel, us):
    national_dc_growth(panel)
    national_electricity(us)
    national_water(us)
    national_holistic(us, panel)
    national_correlations(us)
    state_rankings(panel)
    state_growth_curves(panel)
    state_holistic(panel)
    state_resource_scatter(panel)
