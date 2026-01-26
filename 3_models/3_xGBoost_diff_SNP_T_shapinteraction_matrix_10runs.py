#%%
# Import necessary libraries
import os
import pandas as pd
import numpy as np
import sys
from sklearn.impute import SimpleImputer 
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, RepeatedKFold
from joblib import dump, load
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import shap
import datatable as dt 
import glob

# %%
# Set up directories for outputs
models_dir = "models/GB_ap13/shapinter_models"
results_dir = "models/GB_ap13/shapinter_models/shapinter_results/all_imp_features"
feature_importance = "feature_importance/shapinter_models"
shap_values = "shap_values/shapinter_models/all_imp_features"

# Create output directories if they don't exist
for directory in [models_dir, results_dir, feature_importance, shap_values]:
    os.makedirs(directory, exist_ok=True)

#%%
# Set up trait and environment
traits = range(0, 6)
job_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
trait_number = traits[job_id]
env = "tx"  # Choose from "tx", "mi", or "diff"

# Load TPM (expression) data depending on environment
if env == "tx":
    data_tpm = pd.read_csv("../data/ML/tpm_tx.csv", index_col=0)
elif env == "mi":
    data_tpm = pd.read_csv("../data/ML/tpm_mi.csv", index_col=0)
elif env == "diff":
    data_tpm = pd.read_csv("../data/ML/tpm_tx.csv", index_col=0) - pd.read_csv("../data/ML/tpm_mi.csv", index_col=0)

#%%
# Load SNP data
snps = dt.fread("../data/ML/snps.csv").to_pandas()
snps.set_index('PLANT_ID', inplace=True)

#%%
# Load phenotype data based on environment
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
imputer = SimpleImputer(strategy="median")
y_train = pd.DataFrame(imputer.fit_transform(pheno_trn), columns=pheno_trn.columns, index=pheno_trn.index)
y_test = pd.DataFrame(imputer.transform(pheno_tst), columns=pheno_tst.columns, index=pheno_tst.index)
column_name = y_train.columns[trait_number]

#%%
# Load prior feature importance to filter feature space
pattern = os.path.join(feature_importance, f"{column_name}_GB_featureimportance_SNPTPM_{env}_shapinter_perc1.csv")
feature_im_cv = pd.read_csv(glob.glob(pattern)[0], index_col=0)

#%%
# If fewer than 5000 features, use full set
if feature_im_cv.shape[0] < 5000:
    # Merge SNP and TPM data for training/testing
    x_trn = pd.concat([snps.loc[y_train.index], data_tpm.loc[y_train.index]], axis=1)
    x_tst = pd.concat([snps.loc[y_test.index], data_tpm.loc[y_test.index]], axis=1)

    # Subset to selected features
    X_train = x_trn[feature_im_cv.index]
    X_test = x_tst[feature_im_cv.index]

    # Load pre-trained model
    model = load(glob.glob(os.path.join(models_dir, f"{column_name}_GB_SNPTPM_{env}_shapinter_perc1.joblib"))[0])

    # Prepare feature importance tracking
    best_params = model.get_params()
    best_params.pop('random_state', None)
    best_params.pop('n_jobs', None)
    feature_im = pd.DataFrame(index=X_train.columns, columns=range(10))

    #%%
    # Train and interpret with multiple seeds
    for i in range(10):
        new_model = XGBRegressor(n_jobs=-1, random_state=i, **best_params)
        new_model.fit(X_train, y_train[column_name])
        feature_im[i] = new_model.feature_importances_

        explainer = shap.TreeExplainer(new_model)
        shap_vals = explainer(X_train)
        shap_df = pd.DataFrame(shap_vals.values, index=X_train.index, columns=X_train.columns)
        shap_df.to_csv(os.path.join(shap_values, f'{column_name}_GB_shap_values_{env}_SNPT_shapinter_allfeatIm_run{i}.csv'))

        interaction = explainer.shap_interaction_values(X_train)
        abs_interaction = np.abs(interaction)

        for name, data in {
            "mean": np.mean(interaction, axis=0),
            "sum": np.sum(interaction, axis=0),
            "sd": np.std(interaction, axis=0),
            "absmean": np.mean(abs_interaction, axis=0),
            "abssum": np.sum(abs_interaction, axis=0),
            "abssd": np.std(abs_interaction, axis=0)
        }.items():
            df = pd.DataFrame(data, index=X_train.columns, columns=X_train.columns)
            df.to_csv(os.path.join(shap_values, f'{column_name}_GB_{env}_SNPT_{name}_shapinter_allfeatIm_run{i}.csv'))

    # Save feature importance stats
    feature_im['Mean'] = feature_im.mean(axis=1)
    feature_im['SD'] = feature_im.std(axis=1)
    feature_im.to_csv(os.path.join(feature_importance, f'{column_name}_GB_featureimportance_{env}_SNPT_allfeatIm_shapinter.csv'))

else:
    # Use only top 5,000 features
    feature_im_cv = feature_im_cv.sort_values(by='Mean', ascending=False).iloc[:5000]

    x_trn = pd.concat([snps.loc[y_train.index], data_tpm.loc[y_train.index]], axis=1)
    x_tst = pd.concat([snps.loc[y_test.index], data_tpm.loc[y_test.index]], axis=1)

    X_train = x_trn[feature_im_cv.index]
    X_test = x_tst[feature_im_cv.index]

    # Define hyperparameter search space for XGBoost
    space = {
        'learning_rate': hp.uniform('learning_rate', 0.001, 0.3),
        'max_depth': hp.choice('max_depth', [3, 5, 10]),
        'subsample': hp.uniform('subsample', 0.8, 0.9),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200])
    }

    # Objective function for HyperOpt
    def objective(params):
        model = XGBRegressor(**params, random_state=42, n_jobs=-1)
        score = -np.mean(cross_val_score(model, X_train, y_train[column_name], cv=5, scoring='r2', n_jobs=-1))
        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()
    best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
    best_params['max_depth'] = [3, 5, 10][int(best_params['max_depth'])]
    best_params['n_estimators'] = [50, 100, 150, 200][int(best_params['n_estimators'])]

    # Train final model and save results
    def fit_and_save_model(best_params, X_train, y_train, X_test, y_test, column_name):
        model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train[column_name])
        dump(model, os.path.join(models_dir, f'{column_name}_GB_SNPT_{env}_shapinter_5k.joblib'))

        cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
        cv_r2 = cross_val_score(model, X_train, y_train[column_name], cv=cv, scoring='r2', n_jobs=-1)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test[column_name], y_pred)
        pearson_corr, _ = pearsonr(y_test[column_name], y_pred)

        pd.DataFrame({
            "CV_R2_Mean": [cv_r2.mean()],
            "CV_R2_Std": [cv_r2.std()],
            "Test_R2": [r2],
            "Test_pearson_cor": [pearson_corr]
        }).to_csv(os.path.join(results_dir, f'{column_name}_GB_metrics_SNPT_{env}_shapinter_perc5k.csv'), index=False)

    fit_and_save_model(best_params, X_train, y_train, X_test, y_test, column_name)

    # Load final model and repeat SHAP analysis
    model = load(os.path.join(models_dir, f'{column_name}_GB_SNPT_{env}_shapinter_5k.joblib'))

    best_params = model.get_params()
    best_params.pop('random_state', None)
    best_params.pop('n_jobs', None)
    feature_im = pd.DataFrame(index=X_train.columns, columns=range(10))

    for i in range(10):
        new_model = XGBRegressor(n_jobs=-1, random_state=i, **best_params)
        new_model.fit(X_train, y_train[column_name])
        feature_im[i] = new_model.feature_importances_

        explainer = shap.TreeExplainer(new_model)
        shap_df = pd.DataFrame(explainer(X_train).values, index=X_train.index, columns=X_train.columns)
        shap_df.to_csv(os.path.join(shap_values, f'{column_name}_GB_shap_values_SNPT_{env}_shapinter_allfeatIm_run{i}.csv'))

        interaction = explainer.shap_interaction_values(X_train)
        abs_interaction = np.abs(interaction)

        for name, data in {
            "mean": np.mean(interaction, axis=0),
            "sum": np.sum(interaction, axis=0),
            "sd": np.std(interaction, axis=0),
            "absmean": np.mean(abs_interaction, axis=0),
            "abssum": np.sum(abs_interaction, axis=0),
            "abssd": np.std(abs_interaction, axis=0)
        }.items():
            df = pd.DataFrame(data, index=X_train.columns, columns=X_train.columns)
            df.to_csv(os.path.join(shap_values, f'{column_name}_GB_{env}_SNPT_{name}_shapinter_allfeatIm_run{i}.csv'))

    feature_im['Mean'] = feature_im.mean(axis=1)
    feature_im['SD'] = feature_im.std(axis=1)
    feature_im.to_csv(os.path.join(feature_importance, f'{column_name}_GB_featureimportance_{env}_SNPT_allfeatIm_shapinter.csv'))
