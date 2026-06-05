"""
Traffic Demand Prediction Pipeline - Advanced Ensemble

This module loads the dataset, performs advanced feature engineering 
(including historical aggregations, cyclical features, and spatial clustering),
trains an Optuna-optimized 5-model ensemble (XGB, HGB, CatBoost, RF, ET),
and outputs a highly optimized submission file.
"""

import warnings
import pandas as pd
import numpy as np
import pygeohash as pgh
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)


def load_data():
    """Load train, test, and sample submission data."""
    train = pd.read_csv("/Users/anushka/Downloads/dataset/train.csv")
    test = pd.read_csv("/Users/anushka/Downloads/dataset/test.csv")
    return train, test


def feature_engineering(all_data, hist_demand, global_mean_demand):
    """Perform advanced feature engineering on the provided dataframe."""
    df = all_data.copy()
    
    # Merge historical demand
    df = pd.merge(df, hist_demand, on="geohash", how="left")
    df["hist_mean_demand"] = df["hist_mean_demand"].fillna(global_mean_demand)

    # 1. Spatial Features
    df["latitude"] = df["geohash"].apply(
        lambda x: pgh.decode(x)[0] if pd.notnull(x) else np.nan
    )
    df["longitude"] = df["geohash"].apply(
        lambda x: pgh.decode(x)[1] if pd.notnull(x) else np.nan
    )
    
    spatial_coords = df[["latitude", "longitude"]].fillna(df[["latitude", "longitude"]].mean())
    kmeans = KMeans(n_clusters=10, random_state=SEED, n_init=10)
    df["spatial_cluster"] = kmeans.fit_predict(spatial_coords)

    # 2. Time features
    df[["hour", "minute"]] = df["timestamp"].str.split(":", expand=True).astype(float)
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]
    
    # Cyclical Transformations
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute_of_day"] / 1440)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute_of_day"] / 1440)

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
    df["Temp_per_Lane"] = df["Temperature"] / (df["NumberofLanes"] + 2)
    df["Weather_x_Rush"] = df["Weather"] + "_" + df["is_rush_hour"].astype(str)

    # Frequency Encoding
    for col in ["geohash", "Weather", "RoadType"]:
        freq = df[col].value_counts() / len(df)
        df[col + "_freq"] = df[col].map(freq)

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
    for col in cat_cols:
        df[col] = df[col].astype(str)

    return df, cat_cols


def run_pipeline():
    """Run the entire advanced modeling pipeline."""
    print("Loading data...")
    train, test = load_data()

    print("Extracting historical target aggregates...")
    day48_train = train[train["day"] == 48]
    hist_demand = day48_train.groupby("geohash")["demand"].mean().reset_index()
    hist_demand.rename(columns={"demand": "hist_mean_demand"}, inplace=True)
    global_mean_demand = day48_train["demand"].mean()

    print("Feature Engineering...")
    train["is_train"] = 1
    test["is_train"] = 0
    all_data = pd.concat([train, test], axis=0, ignore_index=True)

    all_data, cat_cols = feature_engineering(all_data, hist_demand, global_mean_demand)

    train_df = all_data[all_data["is_train"] == 1].drop(columns=["is_train"])
    test_df = all_data[all_data["is_train"] == 0].drop(columns=["is_train", "demand"])

    features = [
        col for col in train_df.columns if col not in ["Index", "demand", "timestamp"]
    ]

    x_feat = train_df[features].copy()
    y_target = train_df["demand"]
    x_test = test_df[features].copy()
    
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    x_feat[cat_cols] = oe.fit_transform(x_feat[cat_cols])
    x_test[cat_cols] = oe.transform(x_test[cat_cols])
    
    # Convert encoded variables to integers strictly in pandas
    for c in cat_cols:
        x_feat[c] = pd.to_numeric(x_feat[c], downcast='integer')
        x_test[c] = pd.to_numeric(x_test[c], downcast='integer')

    print("Starting K-Fold modeling...")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    def objective_xgb(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": SEED,
            "n_estimators": 200,
            "enable_categorical": True,
            "tree_method": "hist",
        }

        r2_scores = []
        for train_idx, val_idx in kf.split(x_feat, y_target):
            x_tr, y_tr = x_feat.iloc[train_idx], y_target.iloc[train_idx]
            x_va, y_va = x_feat.iloc[val_idx], y_target.iloc[val_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
            preds = model.predict(x_va)
            r2_scores.append(r2_score(y_va, preds))

        return np.mean(r2_scores)

    print("Running Optuna for XGBoost...")
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(objective_xgb, n_trials=3)
    best_xgb_params = study_xgb.best_params
    best_xgb_params.update({
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": SEED,
        "n_estimators": 500,
        "enable_categorical": True,
        "tree_method": "hist"
    })
    
    print("Running Optuna for HistGradientBoosting...")
    def objective_hgb(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "max_iter": 200,
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
            "random_state": SEED
        }
        r2_scores = []
        for train_idx, val_idx in kf.split(x_feat, y_target):
            x_tr, y_tr = x_feat.iloc[train_idx], y_target.iloc[train_idx]
            x_va, y_va = x_feat.iloc[val_idx], y_target.iloc[val_idx]
            model = HistGradientBoostingRegressor(**params)
            model.fit(x_tr, y_tr)
            r2_scores.append(r2_score(y_va, model.predict(x_va)))
        return np.mean(r2_scores)

    study_hgb = optuna.create_study(direction="maximize")
    study_hgb.optimize(objective_hgb, n_trials=3)
    best_hgb_params = study_hgb.best_params
    best_hgb_params.update({"max_iter": 500, "random_state": SEED})

    cat_params = {
        "loss_function": "RMSE",
        "eval_metric": "R2",
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": SEED,
        "verbose": 0,
        "iterations": 500,
        "cat_features": cat_cols,
    }
    rf_params = {"n_estimators": 150, "max_depth": 12, "random_state": SEED, "n_jobs": -1}
    et_params = {"n_estimators": 150, "max_depth": 12, "random_state": SEED, "n_jobs": -1}

    oof_preds = {name: np.zeros(len(x_feat)) for name in ["XGB", "HGB", "CAT", "RF", "ET"]}
    test_preds = {name: np.zeros(len(x_test)) for name in ["XGB", "HGB", "CAT", "RF", "ET"]}

    models = {
        "XGB": xgb.XGBRegressor(**best_xgb_params),
        "HGB": HistGradientBoostingRegressor(**best_hgb_params),
        "CAT": CatBoostRegressor(**cat_params),
        "RF": RandomForestRegressor(**rf_params),
        "ET": ExtraTreesRegressor(**et_params)
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(x_feat, y_target)):
        print(f"--- Fold {fold+1} ---")
        x_tr, y_tr = x_feat.iloc[train_idx], y_target.iloc[train_idx]
        x_va, y_va = x_feat.iloc[val_idx], y_target.iloc[val_idx]
        
        for name, model in models.items():
            model.fit(x_tr, y_tr)
            oof_preds[name][val_idx] = model.predict(x_va)
            test_preds[name] += model.predict(x_test) / 5

    r2_scores_dict = {name: r2_score(y_target, oof_preds[name]) for name in models.keys()}
    for name, score in r2_scores_dict.items():
        print(f"{name} OOF R2: {score:.5f}")

    # Ensembling
    weights = {name: max(score, 0) for name, score in r2_scores_dict.items()}
    sum_w = sum(weights.values())
    weights = {name: w / sum_w for name, w in weights.items()}
    print(f"Ensemble Weights: {weights}")

    final_oof = np.zeros(len(x_feat))
    final_test = np.zeros(len(x_test))

    for name in models.keys():
        final_oof += weights[name] * oof_preds[name]
        final_test += weights[name] * test_preds[name]
        
    ens_r2 = r2_score(y_target, final_oof)
    print(f"\\nFINAL ENSEMBLE OOF R2: {ens_r2:.5f}")

    assert final_test.shape[0] == 41778, f"Expected 41778 rows, got {final_test.shape[0]}"
    assert not np.isnan(final_test).any(), "NaNs found in predictions!"

    sub = pd.DataFrame({"Index": test_df["Index"], "demand": final_test})
    sub.to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv successfully! Shape:", sub.shape)


if __name__ == "__main__":
    run_pipeline()
