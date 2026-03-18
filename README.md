# U.S. Data Center Growth and Resource Demand Forecast

Predicts how many data centers will be built across U.S. states through 2030,
and what that means for commercial electricity and water demand.

## What it does

Trains three ML models on 1990-2020 data and forecasts through 2030:

- **DC openings** -- two-stage cascade model (will any open? then how many?)
- **Electricity** -- gradient boosting regression driven by projected DC count
- **Water** -- same approach as electricity

Forecasts run at both state and national level. The rollout is autoregressive,
meaning each year's predictions feed into the next year's features. For
2021-2024, actual electricity data is injected directly since we have it --
that captures the real AI-driven demand surge those years.

## Setup

```bash
pip install -r requirements.txt
```

Data files go in `data/`:
- `datacenters_clean.csv`
- `electricity_clean.csv`
- `water_clean.csv`

Then open `main.ipynb` and run top to bottom.

## Project structure

```
main.ipynb          run this
data/               input CSVs
src/
  panel.py          loads and merges the three datasets into one panel
  features.py       builds lag features, rolling averages, momentum signals
  models.py         model classes, training loop, hyperparameter tuning
  forecast.py       autoregressive rollout for 2021-2030
  eda_plots.py      historical charts (1990-2020)
  forecast_plots.py prediction charts (2021-2030)
viz/
  eda/              charts saved here after running EDA
    national/
    state/
  forecast/         charts saved here after running forecasts
    national/
    state/
```

## Data notes

- DC openings capped at 2020 -- that is the last year with confirmed data
- Electricity data runs to 2024 and is used as a signal feature in the rollout
- Water data covers 2000-2020 and is missing Alaska, Hawaii, and D.C.
- Commercial sector electricity is the right column to use -- that is where
  data centers show up in the EIA data, not industrial

## Model notes

The DC openings model uses a cascade because roughly 60% of state-years have
zero openings. A single regressor trained on all that data would learn to
predict near zero for everything. The cascade separates the binary question
(any DCs?) from the count question (how many?) which fixes that.

Recent years are weighted 3-4x more in training since pre-2010 data looks
nothing like the current cloud/AI market. A model fit equally on 1993 and
2019 would be pulled in the wrong direction.

R2 on the test period 2016-2020:
- DC openings: ~0.34 (count data is noisy, national totals are more reliable)
- Electricity: ~0.99
- Water: ~0.99
