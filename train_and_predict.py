import pandas as pd
import numpy as np
import pygeohash as pgh
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

def load_data():
    train = pd.read_csv('/Users/anushka/Downloads/dataset/train.csv')
    test = pd.read_csv('/Users/anushka/Downloads/dataset/test.csv')
    sample_sub = pd.read_csv('/Users/anushka/Downloads/dataset/sample_submission.csv')
    return train, test, sample_sub

def feature_engineering(df):
    df = df.copy()
    
    # 1. Geohash features
    df['latitude'] = df['geohash'].apply(lambda x: pgh.decode(x)[0] if pd.notnull(x) else np.nan)
    df['longitude'] = df['geohash'].apply(lambda x: pgh.decode(x)[1] if pd.notnull(x) else np.nan)
    
    # 2. Time features
    # Timestamp is like "0:0", "2:15" (Hour:Minute)
    df[['hour', 'minute']] = df['timestamp'].str.split(':', expand=True).astype(float)
    df['minute_of_day'] = df['hour'] * 60 + df['minute']
    
    # Advanced time features
    df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(16, 19))).astype(int)
    df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
    df['is_weekend'] = (df['day'] % 7 >= 5).astype(int)  # Assuming day sequential and week is 7 days
    
    # 3. Handle Missing Values & Categoricals
    df['Temperature'].fillna(df['Temperature'].median(), inplace=True)
    df['RoadType'].fillna('Unknown', inplace=True)
    df['Weather'].fillna('Unknown', inplace=True)
    df['NumberofLanes'].fillna(-1, inplace=True)
    df['LargeVehicles'].fillna('Unknown', inplace=True)
    df['Landmarks'].fillna('Unknown', inplace=True)
    
    # 4. Interaction Features
    df['Lanes_x_Road'] = df['NumberofLanes'].astype(str) + "_" + df['RoadType']
    df['Weather_x_Rush'] = df['Weather'] + "_" + df['is_rush_hour'].astype(str)
    
    # Encoding categorical features
    cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather', 'geohash', 'Lanes_x_Road', 'Weather_x_Rush']
    for c in cat_cols:
        df[c] = df[c].astype('category')
        
    return df

def run_pipeline():
    print("Loading data...")
    train, test, sample_sub = load_data()
    
    print("Feature Engineering...")
    # Combine for consistent encoding
    train['is_train'] = 1
    test['is_train'] = 0
    all_data = pd.concat([train, test], axis=0, ignore_index=True)
    
    all_data = feature_engineering(all_data)
    
    train_df = all_data[all_data['is_train'] == 1].drop(columns=['is_train'])
    test_df = all_data[all_data['is_train'] == 0].drop(columns=['is_train', 'demand'])
    
    features = [c for c in train_df.columns if c not in ['Index', 'demand', 'timestamp']]
    cat_features = [c for c in features if train_df[c].dtype.name == 'category']
    
    X = train_df[features]
    y = train_df['demand']
    X_test = test_df[features]
    
    print("Starting modeling...")
    
    # LightGBM Params (Fixed for speed, we could use optuna but it might take too long in environment. 
    # For competition grade, we run a short optuna or use robust params). Let's use robust params with early stopping.
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'verbose': -1,
        'random_state': SEED,
        'n_estimators': 1500
    }
    
    cat_params = {
        'loss_function': 'RMSE',
        'eval_metric': 'R2',
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': SEED,
        'verbose': 0,
        'iterations': 1500,
        'cat_features': cat_features
    }
    
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': SEED,
        'n_estimators': 1000,
        'enable_categorical': True,
        'tree_method': 'hist'
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    oof_lgb = np.zeros(len(X))
    preds_lgb = np.zeros(len(X_test))
    oof_cat = np.zeros(len(X))
    preds_cat = np.zeros(len(X_test))
    oof_xgb = np.zeros(len(X))
    preds_xgb = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Fold {fold+1}")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # LightGBM
        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(X_train, y_train, 
                      eval_set=[(X_val, y_val)], 
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[val_idx] = model_lgb.predict(X_val)
        preds_lgb += model_lgb.predict(X_test) / 5
        
        # CatBoost
        # Convert category to string for catboost or use them directly if pandas category
        model_cat = CatBoostRegressor(**cat_params)
        model_cat.fit(X_train, y_train, 
                      eval_set=[(X_val, y_val)], 
                      early_stopping_rounds=50, verbose=False)
        oof_cat[val_idx] = model_cat.predict(X_val)
        preds_cat += model_cat.predict(X_test) / 5
        
        # XGBoost
        # xgb handles category directly if enable_categorical=True
        model_xgb = xgb.XGBRegressor(**xgb_params)
        model_xgb.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False) # xgb standard api early stopping is deprecating, using default 1000 trees
        oof_xgb[val_idx] = model_xgb.predict(X_val)
        preds_xgb += model_xgb.predict(X_test) / 5

    r2_lgb = r2_score(y, oof_lgb)
    r2_cat = r2_score(y, oof_cat)
    r2_xgb = r2_score(y, oof_xgb)
    
    print(f"LGB OOF R2: {r2_lgb:.5f}")
    print(f"CAT OOF R2: {r2_cat:.5f}")
    print(f"XGB OOF R2: {r2_xgb:.5f}")
    
    # Simple weighted ensemble based on R2 (or just average if all are good)
    # Give higher weight to better model
    weights = [max(r2_lgb, 0), max(r2_cat, 0), max(r2_xgb, 0)]
    sum_w = sum(weights)
    if sum_w == 0:
        w_lgb, w_cat, w_xgb = 0.33, 0.33, 0.33
    else:
        w_lgb, w_cat, w_xgb = weights[0]/sum_w, weights[1]/sum_w, weights[2]/sum_w
        
    print(f"Ensemble Weights -> LGB: {w_lgb:.2f}, CAT: {w_cat:.2f}, XGB: {w_xgb:.2f}")
    
    oof_ens = w_lgb*oof_lgb + w_cat*oof_cat + w_xgb*oof_xgb
    r2_ens = r2_score(y, oof_ens)
    print(f"Ensemble OOF R2: {r2_ens:.5f}")
    
    final_preds = w_lgb*preds_lgb + w_cat*preds_cat + w_xgb*preds_xgb
    
    # Create submission
    sub = pd.DataFrame({'Index': test_df['Index'], 'demand': final_preds})
    sub.to_csv('/Users/anushka/Desktop/Internship work 2026 summer/Traffic-Demand-Predication/submission.csv', index=False)
    print("Submission saved to submission.csv")
    
if __name__ == '__main__':
    run_pipeline()
