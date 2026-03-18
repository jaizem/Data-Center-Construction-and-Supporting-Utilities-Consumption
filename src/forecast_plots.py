import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

NAT_DIR   = Path("viz/forecast/national")
STATE_DIR = Path("viz/forecast/state")

BLUE    = "#1D4ED8"
ORANGE  = "#EA580C"
GREEN   = "#15803D"
GREY    = "#9CA3AF"
L_BLUE  = "#DBEAFE"
L_ORG   = "#FED7AA"
L_GREEN = "#DCFCE7"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130,
})

# milestones for the forecast charts (post-2020 ones included here)
FORECAST_MILESTONES = [
    (2006, "AWS launches",       "top"),
    (2012, "Big Data boom",      "bottom"),
    (2015, "Cloud mainstream",   "top"),
    (2017, "GPU cloud training", "bottom"),
    (2020, "COVID remote surge", "top"),
    (2022, "ChatGPT launch",     "bottom"),
    (2023, "AI buildout surge",  "top"),
]


def add_milestones(ax, y_range, domain="hist"):
    # domain='hist' shows pre-2020 milestones, 'both' shows all including AI milestones
    ymin, ymax = y_range
    span = ymax - ymin
    milestones = FORECAST_MILESTONES if domain == "both" else [m for m in FORECAST_MILESTONES if m[0] <= 2020]

    for year, label, position in milestones:
        xmin, xmax = ax.get_xlim()
        if not (xmin <= year <= xmax):
            continue
        color = ORANGE if year > 2020 else GREY
        ax.axvline(year, color=color, lw=0.8, ls="--", alpha=0.5, zorder=1)
        y = ymax - span * 0.04 if position == "top" else ymin + span * 0.03
        ax.text(year + 0.2, y, label, fontsize=7, color=color,
                rotation=90, va="top" if position == "top" else "bottom",
                alpha=0.85, zorder=2)


def _divider(ax):
    ax.axvline(2020.5, color="black", lw=1.2, ls=":", alpha=0.5)


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


# model evaluation charts

def model_fit(y_true, y_pred, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.35, s=20, color=BLUE)
    lim = max(float(np.nanmax(y_true)), float(np.nanmax(y_pred))) * 1.05
    ax.plot([0, lim], [0, lim], "--", color=ORANGE, lw=1.5, label="perfect fit")
    r2  = 1 - np.nansum((y_true - y_pred)**2) / np.nansum((y_true - np.nanmean(y_true))**2)
    mae = np.nanmean(np.abs(y_true - y_pred))
    ax.text(0.05, 0.90, f"R2 = {r2:.3f}\nMAE = {mae:.2f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted", fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[1]
    residuals = np.array(y_pred) - np.array(y_true)
    ax.hist(residuals, bins=35, color=BLUE, alpha=0.7, edgecolor="white")
    ax.axvline(0, color=ORANGE, lw=2, ls="--")
    ax.axvline(np.median(residuals), color=GREY, lw=1.5, ls=":",
               label=f"Median residual: {np.median(residuals):.2f}")
    ax.set_xlabel("Residual (Pred minus Actual)")
    ax.set_ylabel("Count")
    ax.set_title("Residuals", fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, NAT_DIR, filename)


def feature_importance(model, feat_names, title, filename):
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=feat_names).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, len(feat_names) * 0.42)))
    colors = [ORANGE if imp[f] > imp.quantile(0.75) else BLUE for f in imp.index]
    ax.barh(imp.index, imp.values, color=colors, alpha=0.85)
    ax.axvline(imp.mean(), color=GREY, lw=1, ls="--", label="Mean importance")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, NAT_DIR, filename)


# national forecast charts

def national_dc_forecast(hist_us, fc_us):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    h = hist_us[["year", "openings"]].copy()
    f = fc_us[["year", "openings"]].copy()
    ax.bar(h["year"], h["openings"], color=BLUE,   alpha=0.7, width=0.8, label="Historical")
    ax.bar(f["year"], f["openings"], color=ORANGE, alpha=0.7, width=0.8, label="Forecast")
    ax.plot(h["year"], h["openings"].rolling(3, min_periods=1).mean(),
            color=BLUE, lw=1.5, ls="--", alpha=0.6, label="3yr hist avg")
    ax.plot(f["year"], f["openings"].rolling(3, min_periods=1).mean(),
            color=ORANGE, lw=1.5, ls="--", alpha=0.6)
    _divider(ax)
    ymax = max(h["openings"].max(), f["openings"].max()) * 1.15
    add_milestones(ax, (0, ymax), domain="both")
    ax.set_title("Annual U.S. DC Openings", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("New Openings")
    ax.legend(fontsize=9)

    ax = axes[1]
    h["cum"] = h["openings"].cumsum()
    base = int(h["cum"].iloc[-1])
    f["cum"] = base + f["openings"].cumsum()
    ax.fill_between(h["year"], h["cum"], color=L_BLUE, alpha=0.9)
    ax.plot(h["year"], h["cum"], color=BLUE, lw=2.5, label="Historical")
    ax.fill_between(f["year"], f["cum"] * 0.82, f["cum"] * 1.18, color=L_ORG, alpha=0.35)
    ax.fill_between(f["year"], f["cum"], color=L_ORG, alpha=0.9)
    ax.plot(f["year"], f["cum"], color=ORANGE, lw=2.5, ls="--", label="Forecast")
    _divider(ax)
    ymax_cum = f["cum"].max() * 1.1
    add_milestones(ax, (0, ymax_cum), domain="both")
    ax.set_title("Cumulative U.S. Data Centers", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Facilities")
    ax.legend(fontsize=9)
    ax.text(2021.5, f["cum"].min() * 0.92, "18% band", fontsize=7, color=ORANGE, alpha=0.7)

    total_new = int(f["openings"].sum())
    fig.suptitle(f"U.S. Data Center Forecast 2021 to 2030  --  ~{total_new:,} new facilities projected",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, NAT_DIR, "dc_forecast")


def national_resource_forecast(hist_us, fc_us):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, col, hcolor, bg, label, unit, div in [
        (axes[0], "electricity_usage_mwh", BLUE,  L_BLUE,  "Electricity", "Billion MWh", 1e9),
        (axes[1], "water_usage_mgal",       GREEN, L_GREEN, "Water",       "Billion Gal", 1e3),
    ]:
        h = hist_us[hist_us["year"] <= 2020][["year", col]].dropna()
        f = fc_us[["year", col]].dropna()
        if f.empty:
            continue

        ax.fill_between(h["year"], h[col] / div, color=bg, alpha=0.9)
        ax.plot(h["year"], h[col] / div, color=hcolor, lw=2.5, label="Historical")
        ax.fill_between(f["year"], f[col] / div * 0.88, f[col] / div * 1.12, color=L_ORG, alpha=0.35)
        ax.fill_between(f["year"], f[col] / div, color=L_ORG, alpha=0.6)
        ax.plot(f["year"], f[col] / div, color=ORANGE, lw=2.5, ls="--", label="Forecast")
        ax.axhline(h[col].median() / div, color=GREY, lw=1, ls=":",
                   label=f"Hist median: {h[col].median()/div:.2f}")
        _divider(ax)
        ymax = max(h[col].max(), f[col].max()) / div * 1.12
        add_milestones(ax, (h[col].min()/div * 0.98, ymax), domain="both")
        ax.set_title(f"U.S. {label} Demand", fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel(unit)
        ax.legend(fontsize=9)

    fig.suptitle("U.S. Resource Demand Forecast Driven by Data Center Growth",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, NAT_DIR, "resource_forecast")


def national_holistic_forecast(hist_us, fc_us):
    h = hist_us[hist_us["year"] >= 2000].copy()
    f = fc_us.copy()

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(h["year"], h["openings"], color=BLUE, alpha=0.6, width=0.8, label="Historical")
    ax1.bar(f["year"], f["openings"], color=ORANGE, alpha=0.7, width=0.8, label="Forecast")
    ax1.plot(h["year"], h["openings"].rolling(3, min_periods=1).mean(),
             color=BLUE, lw=1.5, ls="--", alpha=0.7)
    _divider(ax1)
    ymax1 = max(h["openings"].max(), f["openings"].max()) * 1.15
    add_milestones(ax1, (0, ymax1), domain="both")
    ax1.set_title("Annual DC Openings", fontweight="bold")
    ax1.set_ylabel("New Openings")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    hist_total = hist_us[hist_us["year"] < 2000]["openings"].sum()
    h["cum"] = h["openings"].cumsum() + hist_total
    base = int(hist_us["openings"].sum())
    f_cum = base + f["openings"].cumsum()
    ax2.fill_between(h["year"], h["cum"], color=L_BLUE, alpha=0.9)
    ax2.fill_between(f["year"], f_cum * 0.82, f_cum * 1.18, color=L_ORG, alpha=0.3)
    ax2.fill_between(f["year"], f_cum, color=L_ORG, alpha=0.8)
    ax2.plot(h["year"], h["cum"], color=BLUE, lw=2.5, label="Historical")
    ax2.plot(f["year"], f_cum, color=ORANGE, lw=2.5, ls="--", label="Forecast")
    _divider(ax2)
    add_milestones(ax2, (0, f_cum.max() * 1.12), domain="both")
    ax2.set_title("Cumulative Data Centers", fontweight="bold")
    ax2.set_ylabel("Total Facilities")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    h_e = h.dropna(subset=["electricity_usage_mwh"])
    f_e = f.dropna(subset=["electricity_usage_mwh"])
    ax3.fill_between(h_e["year"], h_e["electricity_usage_mwh"] / 1e9, color=L_BLUE, alpha=0.9)
    ax3.plot(h_e["year"], h_e["electricity_usage_mwh"] / 1e9, color=BLUE, lw=2.5, label="Historical")
    if not f_e.empty:
        ax3.fill_between(f_e["year"], f_e["electricity_usage_mwh"] / 1e9 * 0.9,
                         f_e["electricity_usage_mwh"] / 1e9 * 1.1, color=L_ORG, alpha=0.35)
        ax3.fill_between(f_e["year"], f_e["electricity_usage_mwh"] / 1e9, color=L_ORG, alpha=0.8)
        ax3.plot(f_e["year"], f_e["electricity_usage_mwh"] / 1e9, color=ORANGE, lw=2.5, ls="--", label="Forecast")
    _divider(ax3)
    ymax3 = max(h_e["electricity_usage_mwh"].max(), f_e["electricity_usage_mwh"].max() if not f_e.empty else 0) / 1e9 * 1.1
    add_milestones(ax3, (h_e["electricity_usage_mwh"].min()/1e9 * 0.98, ymax3), domain="both")
    ax3.set_title("Commercial Electricity Demand", fontweight="bold")
    ax3.set_ylabel("Billion MWh")
    ax3.set_xlabel("Year")
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 1])
    h_w = h.dropna(subset=["water_usage_mgal"])
    f_w = f.dropna(subset=["water_usage_mgal"])
    if not h_w.empty:
        ax4.fill_between(h_w["year"], h_w["water_usage_mgal"] / 1e3, color=L_GREEN, alpha=0.9)
        ax4.plot(h_w["year"], h_w["water_usage_mgal"] / 1e3, color=GREEN, lw=2.5, label="Historical")
    if not f_w.empty:
        ax4.fill_between(f_w["year"], f_w["water_usage_mgal"] / 1e3 * 0.9,
                         f_w["water_usage_mgal"] / 1e3 * 1.1, color=L_ORG, alpha=0.35)
        ax4.fill_between(f_w["year"], f_w["water_usage_mgal"] / 1e3, color=L_ORG, alpha=0.8)
        ax4.plot(f_w["year"], f_w["water_usage_mgal"] / 1e3, color=ORANGE, lw=2.5, ls="--", label="Forecast")
    _divider(ax4)
    ax4.set_title("Water Usage", fontweight="bold")
    ax4.set_ylabel("Billion Gallons")
    ax4.set_xlabel("Year")
    ax4.legend(fontsize=8)

    fig.suptitle("National View -- Data Centers, Electricity and Water  2000 to 2030",
                 fontweight="bold", fontsize=13, y=1.01)
    save(fig, NAT_DIR, "holistic_forecast")


# state forecast charts

def state_dc_forecast(hist_panel, fc_state, top_states):
    ncols = 3
    nrows = -(-len(top_states) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
    axes = axes.flatten()

    for i, state in enumerate(top_states):
        ax = axes[i]
        h  = hist_panel[(hist_panel["state_abbrev"] == state) & (hist_panel["year"] <= 2020)].sort_values("year")
        fc = fc_state[fc_state["state_abbrev"] == state].sort_values("year")

        ax.fill_between(h["year"], h["cum_dcs"], color=L_BLUE, alpha=0.8)
        ax.plot(h["year"], h["cum_dcs"], color=BLUE, lw=2, label="Historical")

        if not fc.empty:
            ax.fill_between(fc["year"], fc["cum_dcs"] * 0.80, fc["cum_dcs"] * 1.20,
                            color=L_ORG, alpha=0.45)
            ax.fill_between(fc["year"], fc["cum_dcs"], color=L_ORG, alpha=0.7)
            ax.plot(fc["year"], fc["cum_dcs"], color=ORANGE, lw=2.5, ls="--", label="Forecast")

        _divider(ax)
        ymax = fc["cum_dcs"].max() * 1.12 if not fc.empty else h["cum_dcs"].max() * 1.1
        add_milestones(ax, (0, ymax), domain="both")
        ax.set_title(state, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Cumulative DCs")
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("State Level Cumulative Data Center Forecast", fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, STATE_DIR, "dc_forecast_grid")


def state_holistic_forecast(hist_panel, fc_state, states=None):
    if states is None:
        states = (
            hist_panel[hist_panel["year"] <= 2020]
            .groupby("state_abbrev")["openings"].sum()
            .sort_values(ascending=False).head(6).index.tolist()
        )

    for state in states:
        h  = hist_panel[
            (hist_panel["state_abbrev"] == state) &
            (hist_panel["year"] >= 2000) &
            (hist_panel["year"] <= 2020)
        ].sort_values("year")
        fc = fc_state[fc_state["state_abbrev"] == state].sort_values("year")

        fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

        ax = axes[0]
        ax.bar(h["year"],  h["openings"],  color=BLUE,   alpha=0.6, width=0.8, label="Historical")
        ax.bar(fc["year"], fc["openings"], color=ORANGE, alpha=0.7, width=0.8, label="Forecast")
        _divider(ax)
        ymax0 = max(h["openings"].max(), fc["openings"].max() if not fc.empty else 0) * 1.15
        add_milestones(ax, (0, ymax0), domain="both")
        ax.set_title(f"{state} -- Data Centers, Electricity and Water  Historical and Forecast", fontweight="bold")
        ax.set_ylabel("Annual DC Openings")
        ax.legend(fontsize=8)

        ax = axes[1]
        h_e  = h.dropna(subset=["electricity_usage_mwh"])
        fc_e = fc.dropna(subset=["electricity_usage_mwh"])
        ax.fill_between(h_e["year"], h_e["electricity_usage_mwh"] / 1e6, color=L_BLUE, alpha=0.8)
        ax.plot(h_e["year"], h_e["electricity_usage_mwh"] / 1e6, color=BLUE, lw=2, label="Historical")
        if not fc_e.empty:
            ax.fill_between(fc_e["year"], fc_e["electricity_usage_mwh"] / 1e6 * 0.9,
                            fc_e["electricity_usage_mwh"] / 1e6 * 1.1, color=L_ORG, alpha=0.3)
            ax.fill_between(fc_e["year"], fc_e["electricity_usage_mwh"] / 1e6, color=L_ORG, alpha=0.7)
            ax.plot(fc_e["year"], fc_e["electricity_usage_mwh"] / 1e6, color=ORANGE, lw=2, ls="--", label="Forecast")
        _divider(ax)
        ax.set_ylabel("Commercial Electricity (M MWh)")
        ax.legend(fontsize=8)

        ax = axes[2]
        h_w  = h.dropna(subset=["water_usage_mgal"])
        fc_w = fc.dropna(subset=["water_usage_mgal"])
        if not h_w.empty:
            ax.fill_between(h_w["year"], h_w["water_usage_mgal"] / 1e3, color=L_GREEN, alpha=0.8)
            ax.plot(h_w["year"], h_w["water_usage_mgal"] / 1e3, color=GREEN, lw=2, label="Historical")
        if not fc_w.empty:
            ax.fill_between(fc_w["year"], fc_w["water_usage_mgal"] / 1e3 * 0.9,
                            fc_w["water_usage_mgal"] / 1e3 * 1.1, color=L_ORG, alpha=0.3)
            ax.fill_between(fc_w["year"], fc_w["water_usage_mgal"] / 1e3, color=L_ORG, alpha=0.7)
            ax.plot(fc_w["year"], fc_w["water_usage_mgal"] / 1e3, color=ORANGE, lw=2, ls="--", label="Forecast")
        _divider(ax)
        ax.set_ylabel("Water Usage (B gallons)")
        ax.set_xlabel("Year")
        ax.legend(fontsize=8)

        fig.tight_layout()
        save(fig, STATE_DIR, f"holistic_forecast_{state.lower()}")


def summary_table_2030(fc_state):
    fc_2030 = fc_state[fc_state["year"] == 2030].copy()
    fc_2030["elec_TWh"]   = (fc_2030["electricity_usage_mwh"] / 1e6).round(1)
    fc_2030["water_Bgal"] = (fc_2030["water_usage_mgal"] / 1e3).round(2)
    top = fc_2030.sort_values("cum_dcs", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    tdata   = top[["state_abbrev", "cum_dcs", "elec_TWh", "water_Bgal"]].values
    clabels = ["State", "Proj. DCs 2030", "Electricity (TWh)", "Water (B gal)"]

    tbl = ax.table(cellText=tdata, colLabels=clabels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.6)
    for j in range(len(clabels)):
        tbl[0, j].set_facecolor(BLUE)
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(tdata) + 1):
        for j in range(len(clabels)):
            tbl[i, j].set_facecolor("#F1F5F9" if i % 2 == 0 else "white")

    ax.set_title("2030 Forecast -- Top 15 States by Projected Data Centers",
                 fontweight="bold", pad=20)
    fig.tight_layout()
    save(fig, NAT_DIR, "summary_table_2030")


def run_all_forecast_plots(hist_panel, hist_us, fc_state, fc_us, test_dfs, models, DC_FEATS, ELEC_FEATS):
    top_states = (
        hist_panel[hist_panel["year"] <= 2020]
        .groupby("state_abbrev")["openings"].sum()
        .sort_values(ascending=False).head(9).index.tolist()
    )

    model_fit(test_dfs["dc"]["openings"].values,
              test_dfs["dc"]["pred_openings"].values,
              "DC Openings Model -- Test Set 2016 to 2020", "model_fit_dc")

    model_fit(test_dfs["elec"]["electricity_usage_mwh"].values,
              test_dfs["elec"]["pred_elec"].values,
              "Electricity Model -- Test Set", "model_fit_elec")

    model_fit(test_dfs["water"]["water_usage_mgal"].values,
              test_dfs["water"]["pred_water"].values,
              "Water Model -- Test Set", "model_fit_water")

    feature_importance(models["dc"].reg, DC_FEATS,
                       "DC Openings -- What drives the prediction", "feat_imp_dc")

    feature_importance(models["elec"].model, ELEC_FEATS,
                       "Electricity Model Feature Importance", "feat_imp_elec")

    national_dc_forecast(hist_us, fc_us)
    national_resource_forecast(hist_us, fc_us)
    national_holistic_forecast(hist_us, fc_us)
    state_dc_forecast(hist_panel, fc_state, top_states)
    state_holistic_forecast(hist_panel, fc_state)
    summary_table_2030(fc_state)
