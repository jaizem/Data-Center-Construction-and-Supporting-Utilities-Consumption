import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")


DC_PARAMS = dict(max_iter=600, learning_rate=0.025, max_depth=5,
                 min_samples_leaf=4, l2_regularization=0.1, random_state=42)

RESOURCE_PARAMS = dict(max_iter=300, learning_rate=0.05, max_depth=4,
                       min_samples_leaf=3, random_state=42)


class DCOpeningsModel:
    # two-stage: classify whether any DCs open, then regress on count
    # the cascade is needed because ~60% of state-years have zero openings
    def __init__(self, params=None):
        p = params or DC_PARAMS
        clf_p = {k: v for k, v in p.items() if k != "l2_regularization"}
        self.clf = HistGradientBoostingClassifier(**clf_p)
        self.reg = HistGradientBoostingRegressor(**p)

    def fit(self, X, y, sample_weight=None):
        self.clf.fit(X, (y > 0).astype(int), sample_weight=sample_weight)
        pos = y > 0
        w_pos = sample_weight[pos] if sample_weight is not None else None
        self.reg.fit(X[pos], np.log1p(y[pos]), sample_weight=w_pos)
        return self

    def predict(self, X):
        counts = np.expm1(self.reg.predict(X)).clip(0)
        return np.where(self.clf.predict(X) == 1, counts, 0.0)


class ResourceModel:
    def __init__(self, params=None):
        self.model = HistGradientBoostingRegressor(**(params or RESOURCE_PARAMS))

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X).clip(0)


def _sample_weights(years):
    # weight recent years more heavily - they're more representative of current market
    w = np.ones(len(years))
    w[years >= 2015] = 3.0
    w[years >= 2018] = 4.0
    w[(years >= 2010) & (years < 2015)] = 1.5
    return w


def train_all_models(featured, DC_FEATS, ELEC_FEATS, WATER_FEATS, train_cutoff=2016, dc_cutoff=2020):
    # training rows: years up to train_cutoff where we have confirmed DC data
    train = featured[(featured["year"] < train_cutoff) & (featured["year"] <= dc_cutoff)].dropna(subset=DC_FEATS + ["openings"])
    test  = featured[(featured["year"] >= train_cutoff) & (featured["year"] <= dc_cutoff)].dropna(subset=DC_FEATS + ["openings"])

    weights = _sample_weights(train["year"].values)

    dc_model = DCOpeningsModel()
    dc_model.fit(train[DC_FEATS].values, train["openings"].values, sample_weight=weights)

    elec_data = featured[(featured["year"] <= dc_cutoff)].dropna(subset=ELEC_FEATS + ["electricity_usage_mwh"])
    water_data = featured[(featured["year"] >= 2000) & (featured["year"] <= dc_cutoff)].dropna(subset=WATER_FEATS + ["water_usage_mgal"])

    e_tr = elec_data[elec_data["year"] < train_cutoff]
    e_te = elec_data[elec_data["year"] >= train_cutoff]
    w_tr = water_data[water_data["year"] < train_cutoff]
    w_te = water_data[water_data["year"] >= train_cutoff]

    elec_model = ResourceModel()
    elec_model.fit(e_tr[ELEC_FEATS].values, e_tr["electricity_usage_mwh"].values)

    water_model = ResourceModel()
    water_model.fit(w_tr[WATER_FEATS].values, w_tr["water_usage_mgal"].values)

    test = test.copy()
    test["pred_openings"] = dc_model.predict(test[DC_FEATS].values)

    e_te = e_te.copy()
    e_te["pred_elec"] = elec_model.predict(e_te[ELEC_FEATS].values)

    w_te = w_te.copy()
    w_te["pred_water"] = water_model.predict(w_te[WATER_FEATS].values)

    models   = {"dc": dc_model, "elec": elec_model, "water": water_model}
    test_dfs = {"dc": test, "elec": e_te, "water": w_te}
    return models, test_dfs


def tune_model(X, y, n_iter=30):
    grid = {
        "max_iter":          [300, 500, 700],
        "learning_rate":     [0.01, 0.025, 0.04, 0.06],
        "max_depth":         [3, 4, 5, 6],
        "min_samples_leaf":  [3, 5, 8],
        "l2_regularization": [0.0, 0.1, 0.5],
    }
    search = RandomizedSearchCV(
        HistGradientBoostingRegressor(random_state=42),
        grid, n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=4),
        n_jobs=-1, random_state=42, verbose=0,
    )
    search.fit(X, np.log1p(y))
    return search.best_params_, abs(search.best_score_)


def score(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    tag  = f"[{label}] " if label else ""
    print(f"{tag}MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}
