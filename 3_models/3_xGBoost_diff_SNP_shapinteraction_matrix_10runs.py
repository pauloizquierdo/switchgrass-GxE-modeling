#%%
# Import required libraries for data processing, modeling, evaluation, and SHAP analysis
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
import glob

# %%
# Define directories for saving models and results
models_dir = "models/GB_ap13/shapinter_models"
results_dir = "models/GB_ap13/shapinter_models/shapinter_results/all_imp_features"
feature_importance = "feature_importance/shapinter_models"
shap_values = "shap_values/shapinter_models/all_imp_features"

# Create directories if they do not exist
for directory in [models_dir, results_dir, feature_importance, shap_values]:
    os.makedirs(directory, exist_ok=True)

#%%
# Define traits and extract job ID for SLURM array jobs
traits = range(0, 6)
job_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
trait_number = traits[job_id]

#%%
# Read SNP data and set PLANT_ID as index
snps = dt.fread("../data/ML/snps.csv").to_pandas()
snps.set_index('PLANT_ID', inplace=True)

#%%
# Define environment and load corresponding phenotype data
env = "tx"

if env == "tx":
    pheno_trn = pd.read_csv("../data/ML/train_pheno_tx.csv", index_col=0)
    pheno_tst = pd.read_csv("../data/ML/test_pheno_tx.csv", index_col=0)
elif env == "mi":
    pheno_trn = pd.read_csv("../data/ML/train_pheno_mi.csv", index_col=0)
    pheno_tst = pd.read_csv("../data/ML/test_pheno_mi.csv", index_col=0)
elif env == "diff":
    # For difference between tx and mi
    pheno_trn = pd.read_csv("../data/ML/train_pheno_tx.csv", index_col=0) - pd.read_csv("../data/ML/train_pheno_mi.csv", index_col=0)
    pheno_tst = pd.read_csv("../data/ML/test_pheno_tx.csv", index_col=0) - pd.read_csv("../data/ML/test_pheno_mi.csv", index_col=0)

#%%
# Impute missing values in phenotypes using median strategy
imputer = SimpleImputer(strategy="median")
y_trn_apk_imputed = pd.DataFrame(imputer.fit_transform(pheno_trn), columns=pheno_trn.columns, index=pheno_trn.index)
y_tst_apk_imputed = pd.DataFrame(imputer.transform(pheno_tst), columns=pheno_tst.columns, index=pheno_tst.index)

y_train, y_test = y_trn_apk_imputed, y_tst_apk_imputed
column_name = y_train.columns[trait_number]

#%%
# Load precomputed SHAP-based feature importance
file_pattern = os.path.join(feature_importance, f"{column_name}_GB_featureimportance_SNP_{env}_shapinter_perc1.csv")
feature_im_cv = pd.read_csv(glob.glob(file_pattern)[0], index_col=0)

# If fewer than 5000 features are available, use them all for model fitting and interpretation
if feature_im_cv.shape[0] < 5000:
    # Filter SNPs by selected features
    X_train = snps.loc[y_train.index, feature_im_cv.index]
    X_test = snps.loc[y_test.index, feature_im_cv.index]

    # Load pre-trained model
    model = load(glob.glob(os.path.join(models_dir, f"{column_name}_GB_SNP_{env}_shapinter_perc1.joblib"))[0])

    # Remove params that will be set manually
    best_params = model.get_params()
    best_params.pop('random_state', None)
    best_params.pop('n_jobs', None)

    feature_im = pd.DataFrame(index=X_train.columns, columns=range(10))

    # Train model with different random seeds to assess feature stability
    for i in range(10):
        new_model = XGBRegressor(n_jobs=-1, random_state=i, **best_params)
        new_model.fit(X_train, y_train[column_name])
        feature_im[i] = new_model.feature_importances_

        explainer = shap.TreeExplainer(new_model)
        shap_vals = explainer(X_train)

        shap_df = pd.DataFrame(shap_vals.values, columns=X_train.columns, index=X_train.index)
        shap_df.to_csv(os.path.join(shap_values, f'{column_name}_GB_shap_values_SNP_{env}_shapinter_allfeatIm_run{i}.csv'))

        # Compute and save SHAP interaction values (mean, sum, sd)
        interaction = explainer.shap_interaction_values(X_train)
        for stat_name, stat_func in [("mean", np.mean), ("sum", np.sum), ("sd", np.std)]:
            df = pd.DataFrame(stat_func(interaction, axis=0), index=X_train.columns, columns=X_train.columns)
            df.to_csv(os.path.join(shap_values, f'{column_name}_GB_{env}_SNP_shap_interaction_values_{stat_name}_shapinter_allfeatIm_run{i}.csv'))

        # Repeat for absolute interaction values
        abs_interaction = np.abs(interaction)
        for stat_name, stat_func in [("mean", np.mean), ("sum", np.sum), ("sd", np.std)]:
            df = pd.DataFrame(stat_func(abs_interaction, axis=0), index=X_train.columns, columns=X_train.columns)
            df.to_csv(os.path.join(shap_values, f'{column_name}_GB_{env}_SNP_absshap_interaction_values_{stat_name}_shapinter_allfeatIm_run{i}.csv'))

    # Save feature importance stats
    feature_im['Mean'] = feature_im.mean(axis=1)
    feature_im['SD'] = feature_im.std(axis=1)
    feature_im.to_csv(os.path.join(feature_importance, f'{column_name}_GB_featureimportance_SNP_{env}_allfeatIm_shapinter.csv'))

else:
    # If >5000 features, keep only top 5k for modeling
    feature_im_cv = feature_im_cv.sort_values(by='Mean', ascending=False).iloc[:5000]
    X_train = snps.loc[y_train.index, feature_im_cv.index]
    X_test = snps.loc[y_test.index, feature_im_cv.index]

    # Define search space for hyperparameter optimization using HyperOpt
    space = {
        'learning_rate': hp.uniform('learning_rate', 0.001, 0.3),
        'max_depth': hp.choice('max_depth', [3, 5, 10]),
        'subsample': hp.uniform('subsample', 0.8, 0.9),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200])
    }

    def objective(params):
        model = XGBRegressor(**params, random_state=42, n_jobs=-1)
        score = -np.mean(cross_val_score(model, X_train, y_train[column_name], cv=5, scoring='r2', n_jobs=-1))
        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()
    best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

    # Decode choices
    best_params['max_depth'] = [3, 5, 10][int(best_params['max_depth'])]
    best_params['n_estimators'] = [50, 100, 150, 200][int(best_params['n_estimators'])]

    # Fit and save the model, compute metrics
    def fit_and_save_model(best_params, X_train, y_train, X_test, y_test, column_name):
        model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train[column_name])
        dump(model, os.path.join(models_dir, f'{column_name}_GB_SNP_{env}_shapinter_5k.joblib'))

        cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
        cv_r2_scores = cross_val_score(model, X_train, y_train[column_name], cv=cv, scoring='r2', n_jobs=-1)
        mean_cv_r2 = cv_r2_scores.mean()
        std_cv_r2 = cv_r2_scores.std()

        y_pred = model.predict(X_test)
        r2_test = r2_score(y_test[column_name], y_pred)
        pearson_corr, _ = pearsonr(y_test[column_name], y_pred)

        metrics = pd.DataFrame({
            "CV_R2_Mean": [mean_cv_r2],
            "CV_R2_Std": [std_cv_r2],
            "Test_R2": [r2_test], 
            "Test_pearson_cor": [pearson_corr]
        })
        metrics.to_csv(os.path.join(results_dir, f'{column_name}_GB_metrics_SNP_{env}_shapinter_perc5k.csv'), index=False)

    fit_and_save_model(best_params, X_train, y_train, X_test, y_test, column_name)

    # Reload model and repeat SHAP-based analysis (as done above)
    model = load(os.path.join(models_dir, f'{column_name}_GB_SNP_{env}_shapinter_5k.joblib'))

    best_params = model.get_params()
    best_params.pop('random_state', None)
    best_params.pop('n_jobs', None)

    feature_im = pd.DataFrame(index=X_train.columns, columns=range(10))

    for i in range(10):
        new_model = XGBRegressor(n_jobs=-1, random_state=i, **best_params)
        new_model.fit(X_train, y_train[column_name])
        feature_im[i] = new_model.feature_importances_

        explainer = shap.TreeExplainer(new_model)
        shap_df = pd.DataFrame(explainer(X_train).values, columns=X_train.columns, index=X_train.index)
        shap_df.to_csv(os.path.join(shap_values, f'{column_name}_GB_shap_values_SNP_{env}_shapinter_allfeatIm_run{i}.csv'))

        interaction = explainer.shap_interaction_values(X_train)
        for stat_name, stat_func in [("mean", np.mean), ("sum", np.sum), ("sd", np.std)]:
            df = pd.DataFrame(stat_func(interaction, axis=0), index=X_train.columns, columns=X_train.columns)
            df.to_csv(os.path.join(shap_values, f'{column_name}_GB_{env}_SNP_shap_interaction_values_{stat_name}_shapinter_allfeatIm_run{i}.csv'))

        abs_interaction = np.abs(interaction)
        for stat_name, stat_func in [("mean", np.mean), ("sum", np.sum), ("sd", np.std)]:
            df = pd.DataFrame(stat_func(abs_interaction, axis=0), index=X_train.columns, columns=X_train.columns)
            df.to_csv(os.path.join(shap_values, f'{column_name}_GB_{env}_SNP_absshap_interaction_values_{stat_name}_shapinter_allfeatIm_run{i}.csv'))

    feature_im['Mean'] = feature_im.mean(axis=1)
    feature_im['SD'] = feature_im.std(axis=1)
    feature_im.to_csv(os.path.join(feature_importance, f'{column_name}_GB_featureimportance_SNP_{env}_allfeatIm_shapinter.csv'))
