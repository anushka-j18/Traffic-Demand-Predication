import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """\
# Traffic Demand Prediction - Competition Grade Solution

This notebook presents a comprehensive machine learning pipeline to predict traffic demand based on historical data. The solution is optimized for R² score.
LightGBM was excluded to avoid Mac OS libomp dependency issues, relying heavily on CatBoost and XGBoost instead.

## Table of Contents
1. Exploratory Data Analysis (EDA)
2. Data Preprocessing & Leakage Prevention
3. Feature Engineering
   - Geohash Decoding
   - Advanced Timestamp Features
   - Interaction Features
4. Modeling and Hyperparameter Tuning with Optuna
   - XGBoost
   - CatBoost
5. Ensembling & Final Submission
"""

code_imports = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pygeohash as pgh
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
"""

text_eda = """\
## 1. Exploratory Data Analysis (EDA)
Let's load the data and inspect it. We'll look at the distribution of the target variable and missing values.
"""

code_eda = """\
train = pd.read_csv('/Users/anushka/Downloads/dataset/train.csv')
test = pd.read_csv('/Users/anushka/Downloads/dataset/test.csv')
sample_sub = pd.read_csv('/Users/anushka/Downloads/dataset/sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
display(train.head())

print("Missing values in Train:\\n", train.isnull().sum()[train.isnull().sum() > 0])
"""

text_fe = """\
## 2. Feature Engineering & Preprocessing
To maximize our R² score, we extract multiple features:
- **Spatial:** Latitude and Longitude from the `geohash`.
- **Temporal:** Parse `timestamp` to `hour` and `minute`. Add `is_rush_hour`, `is_night`, and `is_weekend`.
- **Interactions:** Combine `RoadType` and `NumberofLanes` as an infrastructure capacity metric. We also look at weather and rush hour interaction.
"""

code_fe = """\
def feature_engineering(df):
    df = df.copy()
    
    # 1. Geohash features
    df['latitude'] = df['geohash'].apply(lambda x: pgh.decode(x)[0] if pd.notnull(x) else np.nan)
    df['longitude'] = df['geohash'].apply(lambda x: pgh.decode(x)[1] if pd.notnull(x) else np.nan)
    
    # 2. Time features
    df[['hour', 'minute']] = df['timestamp'].str.split(':', expand=True).astype(float)
    df['minute_of_day'] = df['hour'] * 60 + df['minute']
    
    # Advanced time features
    df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(16, 19))).astype(int)
    df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
    df['is_weekend'] = (df['day'] % 7 >= 5).astype(int)
    
    # 3. Handle Missing Values
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())
    df['RoadType'] = df['RoadType'].fillna('Unknown')
    df['Weather'] = df['Weather'].fillna('Unknown')
    df['NumberofLanes'] = df['NumberofLanes'].fillna(-1)
    df['LargeVehicles'] = df['LargeVehicles'].fillna('Unknown')
    df['Landmarks'] = df['Landmarks'].fillna('Unknown')
    
    # 4. Interaction Features
    df['Lanes_x_Road'] = df['NumberofLanes'].astype(str) + "_" + df['RoadType']
    df['Weather_x_Rush'] = df['Weather'] + "_" + df['is_rush_hour'].astype(str)
    
    # Encode categoricals for XGBoost/CatBoost
    cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather', 'geohash', 'Lanes_x_Road', 'Weather_x_Rush']
    for c in cat_cols:
        df[c] = df[c].astype('category')
        
    return df

# Apply feature engineering
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
"""

text_cv = """\
## 3. Cross Validation Setup and Optuna Tuning
We use a 5-fold Cross-Validation strategy. Optuna is utilized to find the optimal hyperparameters for XGBoost.
(Note: For the sake of this notebook's execution time, n_trials is set lower. In a real competition, run ~100 trials).
"""

code_cv = """\
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': SEED,
        'n_estimators': 300,
        'enable_categorical': True,
        'tree_method': 'hist'
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

# Optimize XGBoost
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=3)
print("Best XGB params:", study_xgb.best_params)
"""

text_models = """\
## 4. Final Models Training
We train XGBoost (with best params) and CatBoost on 5 folds to generate Out-Of-Fold (OOF) predictions and test predictions.
"""

code_models = """\
best_xgb_params = study_xgb.best_params
best_xgb_params['objective'] = 'reg:squarederror'
best_xgb_params['eval_metric'] = 'rmse'
best_xgb_params['random_state'] = SEED
best_xgb_params['n_estimators'] = 800
best_xgb_params['enable_categorical'] = True
best_xgb_params['tree_method'] = 'hist'

cat_params = {
    'loss_function': 'RMSE',
    'eval_metric': 'R2',
    'learning_rate': 0.05,
    'depth': 6,
    'random_seed': SEED,
    'verbose': 0,
    'iterations': 1000,
    'cat_features': cat_features
}

oof_xgb = np.zeros(len(X))
preds_xgb = np.zeros(len(X_test))
oof_cat = np.zeros(len(X))
preds_cat = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Training Fold {fold+1}...")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # XGBoost
    model_xgb = xgb.XGBRegressor(**best_xgb_params)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    preds_xgb += model_xgb.predict(X_test) / 5
    
    # CatBoost
    model_cat = CatBoostRegressor(**cat_params)
    model_cat.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    oof_cat[val_idx] = model_cat.predict(X_val)
    preds_cat += model_cat.predict(X_test) / 5

print("XGB R2:", r2_score(y, oof_xgb))
print("CAT R2:", r2_score(y, oof_cat))
"""

text_ensemble = """\
## 5. Ensembling & Submission
We assign weights based on the validation R² scores.
"""

code_ensemble = """\
r2_xgb = r2_score(y, oof_xgb)
r2_cat = r2_score(y, oof_cat)

weights = [max(r2_xgb, 0), max(r2_cat, 0)]
sum_w = sum(weights)
if sum_w == 0:
    w_xgb, w_cat = 0.5, 0.5
else:
    w_xgb, w_cat = weights[0]/sum_w, weights[1]/sum_w

print(f"Weights -> XGB: {w_xgb:.3f}, CAT: {w_cat:.3f}")

oof_ens = w_xgb*oof_xgb + w_cat*oof_cat
print("Ensemble R2:", r2_score(y, oof_ens))

final_preds = w_xgb*preds_xgb + w_cat*preds_cat

sub = pd.DataFrame({'Index': test_df['Index'], 'demand': final_preds})
sub.to_csv('submission.csv', index=False)
print("Saved submission.csv successfully!")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text_eda),
    nbf.v4.new_code_cell(code_eda),
    nbf.v4.new_markdown_cell(text_fe),
    nbf.v4.new_code_cell(code_fe),
    nbf.v4.new_markdown_cell(text_cv),
    nbf.v4.new_code_cell(code_cv),
    nbf.v4.new_markdown_cell(text_models),
    nbf.v4.new_code_cell(code_models),
    nbf.v4.new_markdown_cell(text_ensemble),
    nbf.v4.new_code_cell(code_ensemble)
]

with open('/Users/anushka/Desktop/Internship work 2026 summer/Traffic-Demand-Predication/traffic_demand_prediction.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Jupyter Notebook created successfully!")
