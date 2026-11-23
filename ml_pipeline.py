"""
ml_pipeline.py
--------------
Trains and compares TWO models on the project's real data:
    1. Linear Regression   (simple, interpretable baseline model)
    2. Random Forest Regressor (non-linear, handles interactions)

Because Sales = Units_Sold x Unit_Price exactly in this dataset, two different
questions are modelled so the comparison is honest and meaningful:

  Task A - "Sales (order value)"   target = Sales
           inputs = Product, Region, Month, Units_Sold, Unit_Price
           -> an easy task: the rule is a multiplication. A straight line cannot
              represent a multiplication well, a forest can. Shows model difference.

  Task B - "Demand (units per order)"   target = Units_Sold
           inputs = Product, Region, Month, Day of week   (no price / sales -> no leakage)
           -> a genuinely hard task: the data only has a seasonal (Nov/Dec) signal,
              so scores are modest. This is the real "demand prediction" task.

Run directly (`python train_models.py`) to train and save everything.
"""
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import BASE_DIR

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3
MODEL_FILE = BASE_DIR / "models" / "trained_models.joblib"

TASKS = {
    "sales": dict(
        label="Task A: Sales (order value)",
        target="Sales",
        cat=["Product", "Region", "Month_Name"],
        num=["Units_Sold", "Unit_Price"],
        unit="",
    ),
    "demand": dict(
        label="Task B: Demand (units per order)",
        target="Units_Sold",
        cat=["Product", "Region", "Month_Name", "Day_Name"],
        num=[],
        unit=" units",
    ),
}

LR_NAME = "Linear Regression"
RF_NAME = "Random Forest"


def _preprocessor(cat, num):
    """One-hot encode text columns, scale numeric columns."""
    steps = [("cat", OneHotEncoder(handle_unknown="ignore"), cat)]
    if num:
        steps.append(("num", StandardScaler(), num))
    return ColumnTransformer(steps)


def _metrics(y_true, y_pred):
    return dict(
        R2=r2_score(y_true, y_pred),
        MAE=mean_absolute_error(y_true, y_pred),
        RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


def _feature_importance(rf_pipe, cat, num):
    """Random-Forest importance, summed back to the original input columns."""
    names = rf_pipe.named_steps["prep"].get_feature_names_out()
    imp = rf_pipe.named_steps["model"].feature_importances_
    agg = {}
    for n, v in zip(names, imp):
        n = n.split("__", 1)[1]
        owner = next((c for c in cat if n.startswith(c + "_")), n)
        agg[owner] = agg.get(owner, 0.0) + float(v)
    s = pd.Series(agg).sort_values(ascending=False)
    return s


def train_task(df: pd.DataFrame, key: str) -> dict:
    cfg = TASKS[key]
    cat, num, target = cfg["cat"], cfg["num"], cfg["target"]
    X, y = df[cat + num], df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # --- Model 1: Linear Regression ------------------------------------------------
    lr = Pipeline([("prep", _preprocessor(cat, num)), ("model", LinearRegression())])
    lr_cv = cross_val_score(lr, X_tr, y_tr, cv=CV_FOLDS, scoring="r2").mean()
    lr.fit(X_tr, y_tr)

    # --- Model 2: Random Forest (min_samples_leaf picked by cross-validation on TRAIN only)
    rf = Pipeline([("prep", _preprocessor(cat, num)),
                   ("model", RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))])
    grid = GridSearchCV(rf, {"model__min_samples_leaf": [1, 10, 30]}, cv=CV_FOLDS, scoring="r2")
    grid.fit(X_tr, y_tr)
    rf = grid.best_estimator_
    rf_cv = grid.best_score_
    best_leaf = grid.best_params_["model__min_samples_leaf"]

    # --- Reference: "always predict the average" -----------------------------------------
    base = Pipeline([("prep", _preprocessor(cat, num)), ("model", DummyRegressor(strategy="mean"))]).fit(X_tr, y_tr)

    preds = {LR_NAME: lr.predict(X_te), RF_NAME: rf.predict(X_te), "Baseline (always predict mean)": base.predict(X_te)}
    rows = []
    for name, p in preds.items():
        m = _metrics(y_te, p)
        m["CV_R2"] = lr_cv if name == LR_NAME else rf_cv if name == RF_NAME else np.nan
        m["Model"] = name
        rows.append(m)
    metrics = pd.DataFrame(rows)[["Model", "R2", "MAE", "RMSE", "CV_R2"]]

    real = metrics[metrics["Model"].isin([LR_NAME, RF_NAME])]
    best_name = real.sort_values("R2", ascending=False).iloc[0]["Model"]

    return dict(
        key=key, cfg=cfg,
        models={LR_NAME: lr, RF_NAME: rf},
        best_name=best_name,
        metrics=metrics,
        rf_leaf=best_leaf,
        y_test=y_te.to_numpy(),
        preds={k: v for k, v in preds.items()},
        importance=_feature_importance(rf, cat, num),
        n_train=len(X_tr), n_test=len(X_te),
    )


def train_all(df: pd.DataFrame) -> dict:
    warnings.filterwarnings("ignore")
    out = {k: train_task(df, k) for k in TASKS}
    out["_sklearn_version"] = sklearn.__version__
    return out


def save_models(results: dict, path=MODEL_FILE):
    path.parent.mkdir(exist_ok=True)
    joblib.dump(results, path)


def load_models(path=MODEL_FILE):
    """Return saved results, or None if missing / saved with another scikit-learn version."""
    if not path.exists():
        return None
    try:
        res = joblib.load(path)
        return res if res.get("_sklearn_version") == sklearn.__version__ else None
    except Exception:
        return None


def predict_one(pipeline, row: dict, key: str) -> float:
    cfg = TASKS[key]
    X = pd.DataFrame([row])[cfg["cat"] + cfg["num"]]
    return float(pipeline.predict(X)[0])
