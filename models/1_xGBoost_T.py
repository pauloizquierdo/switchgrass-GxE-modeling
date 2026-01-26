#%%
# Import required libraries for data processing, machine learning, and model interpretation
import os
import pandas as pd
import numpy as np
import itertools
import sys
from sklearn.impute import SimpleImputer 
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, RepeatedKFold
from joblib import dump, Parallel, delayed
from sklearn.metrics import r2_score, make_scorer
from scipy.stats import pearsonr
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from joblib import load
import shap

#%%
# Define trait indices and select trait to model based on SLURM job array
traits = range(0, 6)  # Total number of traits = 6
job_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
trait_number = traits[job_id]

#%%
# Load transcriptomic expression data for specified environment
env = "tx"  # Options: "tx", "mi", or "diff"

if env == "tx":
    data_tpm = pd.read_csv("../data/ML/tpm_tx.csv", index_col=0)
elif env == "mi":
    data_tpm = pd.read_csv("../data/ML/tpm_mi.csv", index_col=0)
elif env == "diff":
    tpm_tx = pd.read_csv("../data/ML/tpm_tx.csv", index_col=0)
    tpm_mi = pd.read_csv("../data/ML/tpm_mi.csv", index_col=0)
    data_tpm = tpm_tx - tpm_mi  # Compute differential expression

#%%
# Load phenotype data matching the same environmental condition
if env == "tx":
    pheno_trn = pd.read_csv("../data/ML/train_pheno_tx.csv", index_col=0)
    pheno_tst = pd.read_csv("../data/ML/test_pheno_tx.csv", index_col=0)
elif env == "mi":
    pheno_trn = pd.read_csv("../data/ML/train_pheno_mi.csv", index_col=0)
    pheno_tst = pd.read_csv("../data/ML/test_pheno_mi.csv", index_col=0)
elif env == "diff":
    pheno_trn_tx = pd.read_csv("../data/ML/train_pheno_tx.csv", index_col=0)
    pheno_trn_mi = pd.read_csv("../data/ML/train_pheno_mi.csv", index_col=0)
    pheno_trn = pheno_trn_tx - pheno_trn_mi

    pheno_tst_tx = pd.read_csv("../data/ML/test_pheno_tx.csv", index_col=0)
    pheno_tst_mi = pd.read_csv("../data/ML/test_pheno_mi.csv", index_col=0)
    pheno_tst = pheno_tst_tx - pheno_tst_mi

#%%
# Impute missing values in phenotype data using median
imputer = SimpleImputer(strategy="median")
imputer.fit(pheno_trn)

y_trn_apk_imputed = pd.DataFrame(imputer.transform(pheno_trn), columns=pheno_trn.columns, index=pheno_trn.index)
y_tst_apk_imputed = pd.DataFrame(imputer.transform(pheno_tst), columns=pheno_tst.columns, index=pheno_tst.index)

#%%
# Filter and align expression data with phenotype indices
x_trn = data_tpm.loc[y_trn_apk_imputed.index]
x_tst = data_tpm.loc[y_tst_apk_imputed.index]

# Define inputs for modeling
X_train = x_trn
X_test = x_tst
y_train = y_trn_apk_imputed
y_test = y_tst_apk_imputed
column_name = y_train.columns[trait_number]  # Select the trait to model

#%%
# Create directories for saving models and results
models_dir = "models/GB_ap13"
results_dir = os.path.join(models_dir, "GB_results")
feature_importance = "feature_importance"
shap_values = "shap_values"

for d in [models_dir, results_dir, feature_importance, shap_values]:
    os.makedirs(d, exist_ok=True)

#%%
# Define hyperparameter space for XGBoost tuning with HyperOpt
space = {
    'learning_rate': hp.uniform('learning_rate', 0.001, 0.3),
    'max_depth': hp.choice('max_depth', [3, 5, 10]),
    'subsample': hp.uniform('subsample', 0.8, 0.9),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200])
}

# Define objective function to minimize (negative R^2)
def objective(params):
    model = XGBRegressor(**params, random_state=42)
    score = -np.mean(cross_val_score(model, X_train, y_train[column_name], cv=5, scoring='r2', n_jobs=-1))
    return {'loss': score, 'status': STATUS_OK}

# Run hyperparameter optimization
trials = Trials()
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

# Convert chosen categorical parameters from index to actual values
best_params['max_depth'] = [3, 5, 10][int(best_params['max_depth'])]
best_params['n_estimators'] = [50, 100, 150, 200][int(best_params['n_estimators'])]

#%%
# Train final model, evaluate performance, and save outputs
def fit_and_save_model(best_params, X_train, y_train, X_test, y_test, column_name):
    model = XGBRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train[column_name])

    dump(model, os.path.join(models_dir, f'{column_name}_GB_{env}.joblib'))

    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    cv_r2_scores = cross_val_score(model, X_train, y_train[column_name], cv=cv, scoring='r2', n_jobs=-1)

    mean_cv_r2 = cv_r2_scores.mean()
    std_cv_r2 = cv_r2_scores.std()

    y_test_pred = model.predict(X_test)
    r2_test = r2_score(y_test[column_name], y_test_pred)
    pearson_corr_test, _ = pearsonr(y_test[column_name], y_test_pred)

    metrics = pd.DataFrame({
        "CV_R2_Mean": [mean_cv_r2],
        "CV_R2_Std": [std_cv_r2],
        "Test_R2": [r2_test], 
        "Test_pearson_cor": [pearson_corr_test]
    })

    metrics.to_csv(os.path.join(results_dir, f'{column_name}_GB_metrics_{env}.csv'), index=False)

fit_and_save_model(best_params, X_train, y_train, X_test, y_test, column_name)

#%%
# Load model and compute feature importance over multiple seeds
model = load(os.path.join(models_dir, f'{column_name}_GB_{env}.joblib'))

n_runs = 10  # Number of seeds to assess feature importance stability
best_params = model.get_params()
best_params.pop('random_state', None)
best_params.pop('n_jobs', None)

feature_im = pd.DataFrame(index=X_train.columns, columns=range(n_runs))

for i in range(n_runs):
    new_model = XGBRegressor(n_jobs=-1, random_state=i, **best_params)
    new_model.fit(X_train, y_train[column_name])
    feature_im[i] = new_model.feature_importances_

# Compute mean and standard deviation of feature importances
feature_im['Mean'] = feature_im.mean(axis=1)
feature_im['SD'] = feature_im.std(axis=1)

feature_im.to_csv(os.path.join(feature_importance, f'{column_name}_GB_featureimportance_{env}.csv'))

#%%
# Compute SHAP values for the final model to assess feature contributions
explainer_model = shap.TreeExplainer(model)
shap_values_model = explainer_model(X_train)

shap_values_df_model = pd.DataFrame(shap_values_model.values, columns=X_train.columns)
shap_values_df_model.index = X_train.index
shap_values_df_model.to_csv(os.path.join(shap_values, f'{column_name}_GB_shap_values_{env}.csv'))