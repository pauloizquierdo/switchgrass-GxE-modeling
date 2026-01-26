#%%
# Import necessary packages for data manipulation, machine learning, and model interpretation
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
# Define traits to model and generate all combinations (datasets x traits)
traits = range(0, 6) # Adjusted to six traits; may update if more traits are used

# Retrieve job ID from SLURM environment variable for parallel execution
job_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
trait_number = traits[job_id]  # Select the trait for this job

#%%
# Load expression data for a specified environmental condition
# Options: "tx" (e.g. treatment), "mi" (control), or "diff" (difference)
env = "tx"  # Change to "mi" or "diff" to switch context

if env == "tx":
    data_tpm = pd.read_csv("../data/ML/tpm_tx.csv", index_col=0)
elif env == "mi":
    data_tpm = pd.read_csv("../data/ML/tpm_mi.csv", index_col=0)
elif env == "diff":
    tpm_tx = pd.read_csv("../data/ML/tpm_tx.csv", index_col=0)
    tpm_mi = pd.read_csv("../data/ML/tpm_mi.csv", index_col=0)
    data_tpm = tpm_tx - tpm_mi  # Differential expression

#%%
# Load SNP genotype data and set PLANT_ID as index
snps = dt.fread("../data/ML/snps.csv")
snps = snps.to_pandas()
snps.set_index('PLANT_ID', inplace=True)

#%%
# Load phenotype data for the selected environment
y_trn_apk_new, y_tst_apk_new = None, None
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
# Impute missing phenotype values using median strategy
imputer = SimpleImputer(strategy="median")
imputer.fit(pheno_trn)

y_trn_apk_imputed = pd.DataFrame(imputer.transform(pheno_trn), columns=pheno_trn.columns, index=pheno_trn.index)
y_tst_apk_imputed = pd.DataFrame(imputer.transform(pheno_tst), columns=pheno_tst.columns, index=pheno_tst.index)

#%%
# Filter and align genotype and expression data by sample IDs in phenotype tables
x_trn_snps = snps.loc[y_trn_apk_imputed.index]
x_trn_tpm = data_tpm.loc[y_trn_apk_imputed.index]

x_tst_snps = snps.loc[y_tst_apk_imputed.index]
x_tst_tpm = data_tpm.loc[y_tst_apk_imputed.index]

#%%
# Concatenate SNP and transcriptomic features
x_trn = pd.concat([x_trn_snps, x_trn_tpm], axis=1)
x_tst = pd.concat([x_tst_snps, x_tst_tpm], axis=1)

#%%
# Define input features and targets for training/testing
X_train = x_trn
X_test = x_tst
y_train = y_trn_apk_imputed
y_test = y_tst_apk_imputed
column_name = y_train.columns[trait_number]  # Select the target trait

#%%
# Prepare directories for output storage
models_dir = "models/GB_ap13"
results_dir = os.path.join(models_dir, "GB_results")
feature_importance = "feature_importance"
shap_values = "shap_values"

for d in [models_dir, results_dir, feature_importance, shap_values]:
    os.makedirs(d, exist_ok=True)

#%%
# Define hyperparameter search space for XGBoost model
space = {
    'learning_rate': hp.uniform('learning_rate', 0.001, 0.3),
    'max_depth': hp.choice('max_depth', [3, 5, 10]),
    'subsample': hp.uniform('subsample', 0.8, 0.9),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200])
}

# Objective function to minimize in hyperparameter tuning
def objective(params):
    model = XGBRegressor(**params, random_state=42)
    score = -np.mean(cross_val_score(model, X_train, y_train[column_name], cv=5, scoring='r2', n_jobs=-1))
    return {'loss': score, 'status': STATUS_OK}

# Run HyperOpt optimization
trials = Trials()
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

# Recover actual values from indices in hp.choice
best_params['max_depth'] = [3, 5, 10][int(best_params['max_depth'])]
best_params['n_estimators'] = [50, 100, 150, 200][int(best_params['n_estimators'])]

#%%
# Fit and evaluate model, save outputs

def fit_and_save_model(best_params, X_train, y_train, X_test, y_test, column_name):
    model = XGBRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train[column_name])
    dump(model, os.path.join(models_dir, f'{column_name}_GB_SNPTPM_{env}.joblib'))

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

    metrics.to_csv(os.path.join(results_dir, f'{column_name}_GB_metrics_SNPTPM_{env}.csv'), index=False)

# Train and evaluate model
fit_and_save_model(best_params, X_train, y_train, X_test, y_test, column_name)

#%%
# Load final trained model and compute feature importance using multiple random seeds
model = load(os.path.join(models_dir, f'{column_name}_GB_SNPTPM_{env}.joblib'))

n_runs = 10  # Run model multiple times to assess feature importance stability
best_params = model.get_params()
best_params.pop('random_state', None)
best_params.pop('n_jobs', None)

feature_im = pd.DataFrame(index=X_train.columns, columns=range(n_runs))

for i in range(n_runs):
    new_model = XGBRegressor(n_jobs=-1, random_state=i, **best_params)
    new_model.fit(X_train, y_train[column_name])
    feature_im[i] = new_model.feature_importances_

# Aggregate feature importances
feature_im['Mean'] = feature_im.mean(axis=1)
feature_im['SD'] = feature_im.std(axis=1)

feature_im.to_csv(os.path.join(feature_importance, f'{column_name}_GB_featureimportance_SNPTPM_{env}.csv'))

#%%
# Calculate SHAP values for model interpretability
explainer_model = shap.TreeExplainer(model)
shap_values_model = explainer_model(X_train)

# Save SHAP values to CSV
shap_values_df_model = pd.DataFrame(shap_values_model.values, columns=X_train.columns)
shap_values_df_model.index = X_train.index
shap_values_df_model.to_csv(os.path.join(shap_values, f'{column_name}_GB_shap_values_SNPTPM_{env}.csv'))