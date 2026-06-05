import pandas as pd
import numpy as np
import pygeohash as pgh
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)


def load_data():
    train = pd.read_csv("/Users/anushka/Downloads/dataset/train.csv")
    test = pd.read_csv("/Users/anushka/Downloads/dataset/test.csv")
    sample_sub = pd.read_csv("/Users/anushka/Downloads/dataset/sample_submission.csv")
    return train, test, sample_sub


def feature_engineering(df):
    df = df.copy()

    # 1. Geohash features
    df["latitude"] = df["geohash"].apply(
        lambda x: pgh.decode(x)[0] if pd.notnull(x) else np.nan
    )
    df["longitude"] = df["geohash"].apply(
        lambda x: pgh.decode(x)[1] if pd.notnull(x) else np.nan
    )

    # 2. Time features
    # Timestamp is like "0:0", "2:15" (Hour:Minute)
    df[["hour", "minute"]] = df["timestamp"].str.split(":", expand=True).astype(float)
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]

    # Advanced time features
    df["is_rush_hour"] = (
        (df["hour"].between(7, 9)) | (df["hour"].between(16, 19))
    ).astype(int)
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)
    df["is_weekend"] = (df["day"] % 7 >= 5).astype(int)

    # 3. Handle Missing Values & Categoricals
    df["Temperature"] = df["Temperature"].fillna(df["Temperature"].median())
    df["RoadType"] = df["RoadType"].fillna("Unknown")
    df["Weather"] = df["Weather"].fillna("Unknown")
    df["NumberofLanes"] = df["NumberofLanes"].fillna(-1)
    df["LargeVehicles"] = df["LargeVehicles"].fillna("Unknown")
    df["Landmarks"] = df["Landmarks"].fillna("Unknown")

    # 4. Interaction Features
    df["Lanes_x_Road"] = df["NumberofLanes"].astype(str) + "_" + df["RoadType"]
    df["Weather_x_Rush"] = df["Weather"] + "_" + df["is_rush_hour"].astype(str)

    # Encoding categorical features
    cat_cols = [
        "RoadType",
        "LargeVehicles",
        "Landmarks",
        "Weather",
        "geohash",
        "Lanes_x_Road",
        "Weather_x_Rush",
    ]
    for c in cat_cols:
        df[c] = df[c].astype("category")

    return df


def run_pipeline():
    print("Loading data...")
    train, test, sample_sub = load_data()

    print("Feature Engineering...")
    # Combine for consistent encoding
    train["is_train"] = 1
    test["is_train"] = 0
    all_data = pd.concat([train, test], axis=0, ignore_index=True)

    all_data = feature_engineering(all_data)

    train_df = all_data[all_data["is_train"] == 1].drop(columns=["is_train"])
    test_df = all_data[all_data["is_train"] == 0].drop(columns=["is_train", "demand"])

    features = [
        c for c in train_df.columns if c not in ["Index", "demand", "timestamp"]
    ]
    cat_features = [c for c in features if train_df[c].dtype.name == "category"]

    X = train_df[features]
    y = train_df["demand"]
    X_test = test_df[features]

    print("Starting modeling...")

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    def objective_xgb(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": SEED,
            "n_estimators": 300,
            "enable_categorical": True,
            "tree_method": "hist",
        }

        r2_scores = []
        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            r2_scores.append(r2_score(y_val, preds))

        return np.mean(r2_scores)

    print("Running Optuna for XGBoost...")
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(objective_xgb, n_trials=3)
    best_xgb_params = study_xgb.best_params
    best_xgb_params["objective"] = "reg:squarederror"
    best_xgb_params["eval_metric"] = "rmse"
    best_xgb_params["random_state"] = SEED
    best_xgb_params["n_estimators"] = 800
    best_xgb_params["enable_categorical"] = True
    best_xgb_params["tree_method"] = "hist"

    cat_params = {
        "loss_function": "RMSE",
        "eval_metric": "R2",
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": SEED,
        "verbose": 0,
        "iterations": 1000,
        "cat_features": cat_features,
    }

    oof_xgb = np.zeros(len(X))
    preds_xgb = np.zeros(len(X_test))
    oof_cat = np.zeros(len(X))
    preds_cat = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Fold {fold+1}")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # XGBoost
        model_xgb = xgb.XGBRegressor(**best_xgb_params)
        model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = model_xgb.predict(X_val)
        preds_xgb += model_xgb.predict(X_test) / 5

        # CatBoost
        model_cat = CatBoostRegressor(**cat_params)
        model_cat.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False,
        )
        oof_cat[val_idx] = model_cat.predict(X_val)
        preds_cat += model_cat.predict(X_test) / 5

    r2_xgb = r2_score(y, oof_xgb)
    r2_cat = r2_score(y, oof_cat)

    print(f"XGB OOF R2: {r2_xgb:.5f}")
    print(f"CAT OOF R2: {r2_cat:.5f}")

    # Simple weighted ensemble based on R2
    weights = [max(r2_xgb, 0), max(r2_cat, 0)]
    sum_w = sum(weights)
    if sum_w == 0:
        w_xgb, w_cat = 0.5, 0.5
    else:
        w_xgb, w_cat = weights[0] / sum_w, weights[1] / sum_w

    print(f"Ensemble Weights -> XGB: {w_xgb:.2f}, CAT: {w_cat:.2f}")

    oof_ens = w_xgb * oof_xgb + w_cat * oof_cat
    r2_ens = r2_score(y, oof_ens)
    print(f"Ensemble OOF R2: {r2_ens:.5f}")

    final_preds = w_xgb * preds_xgb + w_cat * preds_cat

    # Create submission
    sub = pd.DataFrame({"Index": test_df["Index"], "demand": final_preds})
    sub.to_csv(
        "/Users/anushka/Desktop/Internship work 2026 summer/Traffic-Demand-Predication/submission.csv",
        index=False,
    )
    print("Submission saved to submission.csv")


if __name__ == "__main__":
    run_pipeline()
