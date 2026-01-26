#%%
# Import necessary packages for modeling, hyperparameter optimization, and interpretation
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
import datatable as dt 

#%%
# Define directories for saving models, results, and SHAP data
models_dir = "models/GB_ap13/shapinter_models"
results_dir = os.path.join(models_dir, "shapinter_results")
feature_importance = "feature_importance/shapinter_models"
shap_values = "shap_values/shapinter_models"

for d in [models_dir, results_dir, feature_importance, shap_values]:
    os.makedirs(d, exist_ok=True)

#%%
# Define job combinations for traits and feature selection percentages
traits = range(0, 12)
feature_sel_per = range(0, 15)
job_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
trait_number, feature_percen = traits[job_id]  # Extract trait and feature selection job info

#%%
# Load expression data based on environment
env = "tx"
if env == "tx":
    data_tpm = pd.read_csv("../data/ML/tpm_tx.csv", index_col=0)
elif env == "mi":
    data_tpm = pd.read_csv("../data/ML/tpm_mi.csv", index_col=0)
elif env == "diff":
    tpm_tx = pd.read_csv("../data/ML/tpm_tx.csv", index_col=0)
    tpm_mi = pd.read_csv("../data/ML/tpm_mi.csv", index_col=0)
    data_tpm = tpm_tx - tpm_mi

#%%
# Load SNP genotype data
snps = dt.fread("../data/ML/snps.csv").to_pandas()
snps.set_index('PLANT_ID', inplace=True)

#%%
# Load phenotype data
if env == "tx":
    pheno_trn = pd.read_csv("../data/ML/train_pheno_tx.csv", index_col=0)
    pheno_tst = pd.read_csv("../data/ML/test_pheno_tx.csv", index_col=0)
elif env == "mi":
    pheno_trn = pd.read_csv("../data/ML/train_pheno_mi.csv", index_col=0)
    pheno_tst = pd.read_csv("../data/ML/test_pheno_mi.csv", index_col=0)
elif env == "diff":
    pheno_trn = pd.read_csv("../data/ML/train_pheno_tx.csv", index_col=0) - pd.read_csv("../data/ML/train_pheno_mi.csv", index_col=0)
    pheno_tst = pd.read_csv("../data/ML/test_pheno_tx.csv", index_col=0) - pd.read_csv("../data/ML/test_pheno_mi.csv", index_col=0)

#%%
# Impute missing phenotype values
y_trn_apk_new = pheno_trn
y_tst_apk_new = pheno_tst

imputer = SimpleImputer(strategy="median")
imputer.fit(y_trn_apk_new)

y_trn_apk_imputed = pd.DataFrame(imputer.transform(y_trn_apk_new), columns=y_trn_apk_new.columns, index=y_trn_apk_new.index)
y_tst_apk_imputed = pd.DataFrame(imputer.transform(y_tst_apk_new), columns=y_tst_apk_new.columns, index=y_tst_apk_new.index)

y_train = y_trn_apk_imputed
y_test = y_tst_apk_imputed
column_name = y_train.columns[trait_number]

#%%
# Load SHAP-based feature importance to select top features
feature_im_cv = pd.read_csv(os.path.join('feature_importance', f'{column_name}_GB_featureimportance_SNPTPM_{env}.csv'), index_col=0)
feature_im_cv_NZ = feature_im_cv[feature_im_cv['Mean'] > 0].sort_values(by='Mean', ascending=False)
vect_percen = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]
feature_im_cv_NZ = feature_im_cv_NZ.iloc[0:int(len(feature_im_cv_NZ) * vect_percen[feature_percen]), :]

#%%
# Filter and combine SNP and expression features based on selected importance
x_trn = pd.concat([snps.loc[y_trn_apk_imputed.index], data_tpm.loc[y_trn_apk_imputed.index]], axis=1)
x_tst = pd.concat([snps.loc[y_tst_apk_imputed.index], data_tpm.loc[y_tst_apk_imputed.index]], axis=1)

x_trn = x_trn.loc[:, feature_im_cv_NZ.index]
x_tst = x_tst.loc[:, feature_im_cv_NZ.index]

X_train = x_trn
X_test = x_tst

#%%
# Define hyperparameter search space
space = {
    'learning_rate': hp.uniform('learning_rate', 0.001, 0.3),
    'max_depth': hp.choice('max_depth', [3, 5, 10]),
    'subsample': hp.uniform('subsample', 0.8, 0.9),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200])
}

#%%
# HyperOpt objective function
def objective(params):
    model = XGBRegressor(**params, random_state=42, n_jobs=-1)
    score = -np.mean(cross_val_score(model, X_train, y_train[column_name], cv=5, scoring='r2', n_jobs=-1))
    return {'loss': score, 'status': STATUS_OK}

trials = Trials()
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
best_params['max_depth'] = [3, 5, 10][int(best_params['max_depth'])]
best_params['n_estimators'] = [50, 100, 150, 200][int(best_params['n_estimators'])]

#%%
# Train final model and save results
def fit_and_save_model(best_params, X_train, y_train, X_test, y_test, column_name):
    model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train[column_name])

    dump(model, os.path.join(models_dir, f'{column_name}_GB_SNPTPM_{env}_shapinter_perc{vect_percen[feature_percen]}.joblib'))

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

    metrics.to_csv(os.path.join(results_dir, f'{column_name}_GB_metrics_SNPTPM_{env}_shapinter_perc{vect_percen[feature_percen]}.csv'), index=False)

fit_and_save_model(best_params, X_train, y_train, X_test, y_test, column_name)

#%%
# Load model and assess feature importance
model = load(os.path.join(models_dir, f'{column_name}_GB_SNPTPM_{env}_shapinter_perc{vect_percen[feature_percen]}.joblib'))

n_runs = 10
best_params = model.get_params()
best_params.pop('random_state', None)
best_params.pop('n_jobs', None)

feature_im = pd.DataFrame(index=X_train.columns, columns=range(n_runs))
for i in range(n_runs):
    new_model = XGBRegressor(n_jobs=-1, random_state=i, **best_params)
    new_model.fit(X_train, y_train[column_name])
    feature_im[i] = new_model.feature_importances_

feature_im['Mean'] = feature_im.mean(axis=1)
feature_im['SD'] = feature_im.std(axis=1)
feature_im.to_csv(os.path.join(feature_importance, f'{column_name}_GB_featureimportance_SNPTPM_{env}_shapinter_perc{vect_percen[feature_percen]}.csv'))

#%%
# Compute SHAP values for interpretation
explainer_model = shap.TreeExplainer(model)
shap_values_model = explainer_model(X_train)

shap_values_df_model = pd.DataFrame(shap_values_model.values, columns=X_train.columns)
shap_values_df_model.index = X_train.index
shap_values_df_model.to_csv(os.path.join(shap_values, f'{column_name}_GB_shap_values_SNPTPM_{env}_shapinter_perc{vect_percen[feature_percen]}.csv'))
