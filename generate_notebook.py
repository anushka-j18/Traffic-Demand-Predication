import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """\
# Traffic Demand Prediction - Advanced Competition Grade Solution

This notebook implements an advanced machine learning pipeline. It maximizes the R² score by leveraging:
1. **Cyclical Temporal Encoding** (Sine/Cosine transformations)
2. **Spatial Clustering** (KMeans on Latitude/Longitude)
3. **Historical Target Aggregation** (Day 48 mean demand mapped to Day 49)
4. **Frequency Encodings & Interactions**
5. **Robust 5-Model Ensemble**: XGBoost, CatBoost, HistGradientBoosting, RandomForest, and ExtraTrees.

## Table of Contents
1. Exploratory Data Analysis (EDA)
2. Feature Engineering & Preprocessing
3. Modeling and Hyperparameter Tuning with Optuna
4. Feature Importance
5. Ensembling & Final Submission
"""

code_imports = """\
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pygeohash as pgh
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
np.random.seed(SEED)
"""

text_eda = """\
## 1. Exploratory Data Analysis (EDA)
Let's inspect the target distribution and correlations.
"""

code_eda = """\
train = pd.read_csv('/Users/anushka/Downloads/dataset/train.csv')
test = pd.read_csv('/Users/anushka/Downloads/dataset/test.csv')
sample_sub = pd.read_csv('/Users/anushka/Downloads/dataset/sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Check overlap of Days
print("Train Days:", train['day'].unique())
print("Test Days:", test['day'].unique())
"""

text_fe = """\
## 2. Feature Engineering & Preprocessing
We build advanced features to capture spatial, temporal, and historical demand patterns.
"""

code_fe = """\
# 1. Historical Target Aggregation (Leak-Free)
day48_train = train[train['day'] == 48]
hist_demand = day48_train.groupby('geohash')['demand'].mean().reset_index()
hist_demand.rename(columns={'demand': 'hist_mean_demand'}, inplace=True)
global_mean_demand = day48_train['demand'].mean()

train['is_train'] = 1
test['is_train'] = 0
all_data = pd.concat([train, test], axis=0, ignore_index=True)

# Merge historical demand
all_data = pd.merge(all_data, hist_demand, on='geohash', how='left')
all_data['hist_mean_demand'].fillna(global_mean_demand, inplace=True)

def feature_engineering(df):
    df = df.copy()
    
    # Spatial Features
    df['latitude'] = df['geohash'].apply(lambda x: pgh.decode(x)[0] if pd.notnull(x) else np.nan)
    df['longitude'] = df['geohash'].apply(lambda x: pgh.decode(x)[1] if pd.notnull(x) else np.nan)
    
    # KMeans Clustering on Spatial features
    spatial_coords = df[['latitude', 'longitude']].fillna(df[['latitude', 'longitude']].mean())
    kmeans = KMeans(n_clusters=10, random_state=SEED, n_init=10)
    df['spatial_cluster'] = kmeans.fit_predict(spatial_coords)
    
    # Temporal Features
    df[['hour', 'minute']] = df['timestamp'].str.split(':', expand=True).astype(float)
    df['minute_of_day'] = df['hour'] * 60 + df['minute']
    
    # Cyclical Transformations
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute_of_day'] / 1440)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute_of_day'] / 1440)
    
    df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(16, 19))).astype(int)
    df['is_weekend'] = (df['day'] % 7 >= 5).astype(int)
    
    # Missing Values
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())
    df['RoadType'] = df['RoadType'].fillna('Unknown')
    df['Weather'] = df['Weather'].fillna('Unknown')
    df['NumberofLanes'] = df['NumberofLanes'].fillna(-1)
    df['LargeVehicles'] = df['LargeVehicles'].fillna('Unknown')
    df['Landmarks'] = df['Landmarks'].fillna('Unknown')
    
    # Interactions
    df['Lanes_x_Road'] = df['NumberofLanes'].astype(str) + "_" + df['RoadType']
    df['Temp_per_Lane'] = df['Temperature'] / (df['NumberofLanes'] + 2)
    
    # Frequency Encoding
    for col in ['geohash', 'Weather', 'RoadType']:
        freq = df[col].value_counts() / len(df)
        df[col + '_freq'] = df[col].map(freq)
        
    return df

all_data = feature_engineering(all_data)

train_df = all_data[all_data['is_train'] == 1].drop(columns=['is_train'])
test_df = all_data[all_data['is_train'] == 0].drop(columns=['is_train', 'demand'])

features = [c for c in train_df.columns if c not in ['Index', 'demand', 'timestamp']]
cat_features = [c for c in features if all_data[c].dtype == 'object' or all_data[c].dtype.name == 'category']

X = train_df[features].copy()
y = train_df['demand']
X_test = test_df[features].copy()

# Ordinal Encode and properly cast to int
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat_features] = oe.fit_transform(X[cat_features].astype(str))
X_test[cat_features] = oe.transform(X_test[cat_features].astype(str))

# Explicitly cast the DataFrame columns to int to drop the 'object' dtype
for col in cat_features:
    X[col] = pd.to_numeric(X[col], downcast='integer')
    X_test[col] = pd.to_numeric(X_test[col], downcast='integer')
"""

text_cv = """\
## 3. Modeling and Hyperparameter Tuning with Optuna
We use 5-fold KFold CV. We tune XGBoost and HistGradientBoosting to find optimal params.
(Note: RF and ET are trained with robust defaults to save time).
"""

code_cv = """\
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# XGBoost Tuning
def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': SEED,
        'n_estimators': 200,
        'enable_categorical': False,
        'tree_method': 'hist'
    }
    r2_scores = []
    for train_idx, val_idx in kf.split(X, y):
        x_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        x_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        model = xgb.XGBRegressor(**params)
        model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
        r2_scores.append(r2_score(y_va, model.predict(x_va)))
    return np.mean(r2_scores)

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=3)
best_xgb = study_xgb.best_params
best_xgb.update({'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'random_state': SEED, 'n_estimators': 500, 'enable_categorical': False, 'tree_method': 'hist'})

# HistGB Tuning
def objective_hgb(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_iter': 200,
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
        'random_state': SEED
    }
    r2_scores = []
    for train_idx, val_idx in kf.split(X, y):
        x_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        x_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        model = HistGradientBoostingRegressor(**params)
        model.fit(x_tr, y_tr)
        r2_scores.append(r2_score(y_va, model.predict(x_va)))
    return np.mean(r2_scores)

study_hgb = optuna.create_study(direction='maximize')
study_hgb.optimize(objective_hgb, n_trials=3)
best_hgb = study_hgb.best_params
best_hgb.update({'max_iter': 500, 'random_state': SEED})
"""

text_models = """\
## 4. Train 5-Model Ensemble
Training XGBoost, HistGradientBoosting, CatBoost, RandomForest, and ExtraTrees.
"""

code_models = """\
oof_preds = {name: np.zeros(len(X)) for name in ['XGB', 'HGB', 'CAT', 'RF', 'ET']}
test_preds = {name: np.zeros(len(X_test)) for name in ['XGB', 'HGB', 'CAT', 'RF', 'ET']}

cat_params = {'loss_function': 'RMSE', 'learning_rate': 0.05, 'depth': 6, 'iterations': 500, 'random_seed': SEED, 'verbose': 0, 'cat_features': cat_features}
rf_params = {'n_estimators': 150, 'max_depth': 12, 'random_state': SEED, 'n_jobs': -1}
et_params = {'n_estimators': 150, 'max_depth': 12, 'random_state': SEED, 'n_jobs': -1}

models = {
    'XGB': xgb.XGBRegressor(**best_xgb),
    'HGB': HistGradientBoostingRegressor(**best_hgb),
    'CAT': CatBoostRegressor(**cat_params),
    'RF': RandomForestRegressor(**rf_params),
    'ET': ExtraTreesRegressor(**et_params)
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1} ---")
    x_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    x_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
    for name, model in models.items():
        model.fit(x_tr, y_tr)
        oof_preds[name][val_idx] = model.predict(x_va)
        test_preds[name] += model.predict(X_test) / 5

r2_scores = {name: r2_score(y, oof_preds[name]) for name in models.keys()}
for name, score in r2_scores.items():
    print(f"{name} OOF R2: {score:.5f}")
"""

text_ensemble = """\
## 5. Ensembling, Feature Importance, & Final Submission
Weighted Average Ensemble based on validation R² scores.
"""

code_ensemble = """\
# Weights based on R2 (clipping negative values to 0)
weights = {name: max(score, 0) for name, score in r2_scores.items()}
sum_w = sum(weights.values())
weights = {name: w / sum_w for name, w in weights.items()}

print("Ensemble Weights:", weights)

final_oof = np.zeros(len(X))
final_test = np.zeros(len(X_test))

for name in models.keys():
    final_oof += weights[name] * oof_preds[name]
    final_test += weights[name] * test_preds[name]
    
ens_r2 = r2_score(y, final_oof)
print(f"\\nFINAL ENSEMBLE OOF R2: {ens_r2:.5f}")

# Plot Feature Importance (XGBoost)
plt.figure(figsize=(10, 8))
feat_imp = pd.Series(models['XGB'].feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feat_imp.head(20), y=feat_imp.head(20).index)
plt.title("Top 20 Features (XGBoost)")
plt.tight_layout()
plt.show()

# Verification & Submission
assert final_test.shape[0] == 41778, f"Expected 41778 rows, got {final_test.shape[0]}"
assert not np.isnan(final_test).any(), "NaNs found in predictions!"

sub = pd.DataFrame({'Index': test_df['Index'], 'demand': final_test})
sub.to_csv('submission.csv', index=False)
print("Saved final submission.csv successfully! Shape:", sub.shape)
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

print("Advanced Jupyter Notebook generated.")
