#!/usr/bin/env Rscript
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --array=1-6
#SBATCH --job-name=BRR_perc
#SBATCH -A glbrc
#SBATCH --output=BRR_perc%A_%a.txt
#SBATCH --error=BRR_perc_error0%A_%a.txt

# =============================================================================
# Purpose: Run Bayesian Ridge Regression (BRR) genomic prediction models for
#          six traits using three feature sets (T: transcriptomic/TPM,
#          G: SNPs, GT: combined SNP+TPM) across three environments
#          (MI, TX, and their difference DIFF = plasticity).
#          Models are run using: (1) all features, (2) decreasing percentages
#          of top-ranked features from pre-trained Gradient Boosting (GB) models,
#          and (3) the first 5 PCs of each feature matrix.
#          The script is designed to run as a SLURM array job (1 job per trait).
# =============================================================================

rm(list = ls())

# ---- Libraries ---------------------------------------------------------------
library(tidyverse)   # Data manipulation and visualization
library(ggridges)    # Ridge plots (loaded but used downstream)
library(BGLR)        # Bayesian Generalized Linear Regression (BRR model)
library(data.table)  # Fast file reading with fread()

# ---- SLURM array job setup ---------------------------------------------------
# Each array job corresponds to one trait (1–6)
setwd("/mnt/home/izquier7/Documents/shinhan_lab/projects/switchgrass_rna_seq/scripts/common_samples_env/hyperopt/")

job <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID", "1"))

# Expand grid of traits x datasets; select current job's parameters
JOBS <- expand.grid(trait = 1:6, dataset = 1)
dataset <- as.vector(JOBS[job, "dataset"])
trait   <- as.vector(JOBS[job, "trait"])

# ---- Load feature matrices ---------------------------------------------------
# TPM (transcript abundance) matrices for each environment
data_tpm_mi  <- fread(paste0("../data/ML/data/tpm_mi.csv"))
data_tpm_tx <- fread(paste0("../data/ML/data/tpm_tx.csv"))

data_tpm_mi  <- as.data.frame(data_tpm_mi)
data_tpm_tx <- as.data.frame(data_tpm_tx)

# Set PLANT_ID as row names
data_tpm_mi  <- data_tpm_mi  %>% remove_rownames() %>% column_to_rownames(var = "PLANT_ID")
data_tpm_tx <- data_tpm_tx %>% remove_rownames() %>% column_to_rownames(var = "PLANT_ID")

# Compute plasticity (DIFF): phenotypic difference between environments
data_tpm_diff <- data_tpm_tx - data_tpm_mi

# ---- Load and filter SNP matrix ----------------------------------------------
snps <- fread(paste0("../data/ML/snps.csv"))
snps <- as.data.frame(snps) %>% remove_rownames() %>% column_to_rownames(var = "PLANT_ID")

# Remove monomorphic SNPs (no variation across samples)
index <- apply(snps, 2, function(x) length(unique(x)))
snps  <- snps[, index != 1]

# Compute PCA on SNPs (used later for PCA-based BRR models)
pca_snps <- prcomp(snps, center = TRUE, scale. = TRUE)

# ---- Scale all feature matrices ----------------------------------------------
data_tpm_mi  <- scale(data_tpm_mi)
data_tpm_tx <- scale(data_tpm_tx)
data_tpm_diff <- scale(data_tpm_diff)
snps          <- scale(snps)

# Quick sanity checks on dimensions and sample overlap
dim(snps); dim(data_tpm_mi); dim(data_tpm_tx)
length(which(rownames(data_tpm_mi) %in% rownames(data_tpm_tx)))
length(which(rownames(data_tpm_mi) %in% rownames(snps)))

# ---- Load phenotype data -----------------------------------------------------
# Training sets
pheno_mi_trn  <- read.csv("../data/ML/train_pheno_mi.csv",  row.names = 1)
pheno_tx_trn <- read.csv("../data/ML/train_pheno_tx.csv", row.names = 1)

# Testing sets
pheno_mi_tst  <- read.csv("../data/ML/test_pheno_mi.csv",  row.names = 1)
pheno_tx_tst <- read.csv("../data/ML/test_pheno_tx.csv", row.names = 1)

# Select the six target traits
traits_cols <- c("GR1", "EM1", "FL1", "TC_EOS", "HT_PAN_EOS", "logBiomass")
pheno_mi_trn  <- pheno_mi_trn[,  traits_cols]
pheno_mi_tst  <- pheno_mi_tst[,  traits_cols]
pheno_tx_trn <- pheno_tx_trn[, traits_cols]
pheno_tx_tst <- pheno_tx_tst[, traits_cols]

# Plasticity phenotypes: tx − mi difference
pheno_diff_trn <- pheno_tx_trn - pheno_mi_trn
pheno_diff_tst <- pheno_tx_tst - pheno_mi_tst

# ---- Build multi-omic (GT) feature matrices ----------------------------------
# Combine TPM and SNP matrices by matching row names (common samples only)
data_tpm_mi_snps  <- merge(data_tpm_mi,  snps, by = "row.names", all = FALSE) %>%
                      column_to_rownames(var = "Row.names")
data_tpm_tx_snps <- merge(data_tpm_tx, snps, by = "row.names", all = FALSE) %>%
                      column_to_rownames(var = "Row.names")
data_tpm_diff_snps <- merge(data_tpm_diff, snps, by = "row.names", all = FALSE) %>%
                      column_to_rownames(var = "Row.names")

# ---- Helper function: prediction accuracy ------------------------------------
# Computes R² (coefficient of determination) and Pearson correlation
# between observed (y) and predicted (y_pred) values
r2_cor <- function(y, y_pred) {
  ssr <- sum((y_pred - y)^2)       # Residual sum of squares
  sst <- sum((y - mean(y))^2)      # Total sum of squares
  r2  <- 1 - (ssr / sst)
  pcor <- cor(y, y_pred)
  res <- c(r2, pcor)
  names(res) <- c("r2", "pcor")
  return(res)
}

# ---- Assign phenotype matrices for clarity -----------------------------------
Y_trn_mi  <- pheno_mi_trn;  Y_tst_mi  <- pheno_mi_tst
Y_trn_tx <- pheno_tx_trn; Y_tst_tx <- pheno_tx_tst
Y_trn_diff <- pheno_diff_trn; Y_tst_diff <- pheno_diff_tst

# ---- Load pre-computed GB feature importance files ---------------------------
# Transcriptomic (T) feature importance from Gradient Boosting models
feature_tx <- read.csv(paste0("feature_importance/", colnames(Y_trn_mi)[trait], "_GB_featureimportance_txWgs_dataset1.csv"),  row.names = 1, header = TRUE)
feature_mi  <- read.csv(paste0("feature_importance/", colnames(Y_trn_mi)[trait], "_GB_featureimportance_miWgs_dataset1.csv"),   row.names = 1, header = TRUE)
feature_diff <- read.csv(paste0("feature_importance/", colnames(Y_trn_mi)[trait], "_GB_featureimportance_diffWgs_dataset1.csv"),  row.names = 1, header = TRUE)

# SNP (G) feature importance
feature_tx_G <- read.csv(paste0("feature_importance/", colnames(Y_trn_mi)[trait], "_GB_featureimportance_txSnps_dataset1.csv"), row.names = 1, header = TRUE)
feature_mi_G  <- read.csv(paste0("feature_importance/", colnames(Y_trn_mi)[trait], "_GB_featureimportance_miSnps_dataset1.csv"),  row.names = 1, header = TRUE)
feature_diff_G <- read.csv(paste0("feature_importance/", colnames(Y_trn_mi)[trait], "_GB_featureimportance_diffSnps_dataset1.csv"), row.names = 1, header = TRUE)

# Multi-omic (GT: SNP + TPM) feature importance
feature_tx_GT <- read.csv(paste0("feature_importance/", colnames(Y_trn_mi)[trait], "_GB_featureimportance_txSNPTPM_dataset1.csv"), row.names = 1, header = TRUE)
feature_mi_GT  <- read.csv(paste0("feature_importance/", colnames(Y_trn_mi)[trait], "_GB_featureimportance_miSNPTPM_dataset1.csv"),  row.names = 1, header = TRUE)
feature_diff_GT <- read.csv(paste0("feature_importance/", colnames(Y_trn_mi)[trait], "_GB_featureimportance_diffSNPTPM_dataset1.csv"), row.names = 1, header = TRUE)

# ---- Sort features by mean importance and remove zero-importance features ----
# T
feature_tx <- feature_tx[order(rowMeans(feature_tx), decreasing = TRUE), ] %>% .[rowMeans(.) != 0, ]
feature_mi  <- feature_mi[ order(rowMeans(feature_mi),  decreasing = TRUE), ] %>% .[rowMeans(.) != 0, ]
feature_diff <- feature_diff[order(rowMeans(feature_diff), decreasing = TRUE), ] %>% .[rowMeans(.) != 0, ]

# G
feature_tx_G <- feature_tx_G[order(rowMeans(feature_tx_G), decreasing = TRUE), ] %>% .[rowMeans(.) != 0, ]
feature_mi_G  <- feature_mi_G[ order(rowMeans(feature_mi_G),  decreasing = TRUE), ] %>% .[rowMeans(.) != 0, ]
feature_diff_G <- feature_diff_G[order(rowMeans(feature_diff_G), decreasing = TRUE), ] %>% .[rowMeans(.) != 0, ]

# GT
feature_tx_GT <- feature_tx_GT[order(rowMeans(feature_tx_GT), decreasing = TRUE), ] %>% .[rowMeans(.) != 0, ]
feature_mi_GT  <- feature_mi_GT[ order(rowMeans(feature_mi_GT),  decreasing = TRUE), ] %>% .[rowMeans(.) != 0, ]
feature_diff_GT <- feature_diff_GT[order(rowMeans(feature_diff_GT), decreasing = TRUE), ] %>% .[rowMeans(.) != 0, ]

# ---- MCMC settings for BGLR --------------------------------------------------
nIter  <- 12000   # Total number of MCMC iterations
burnIn <- 2000    # Burn-in iterations to discard

# Percentages of top-ranked features to test (from 100% down to 1%)
runs <- c(1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01)

# =============================================================================
# SECTION 1: Transcriptomic (T) BRR models
# =============================================================================

# Initialize results matrix: rows = all feature%, cols = r2 + Pearson cor
# 48 rows = 1 (all features) + 15 percentages, for 3 environments (mi/tx/DIFF)
accuracy_T <- matrix(0, nrow = 48, ncol = 2)
colnames(accuracy_T) <- c("r2", "pCor")
rownames(accuracy_T) <- c(
  "mi_fetall",  "mi_fet1",  "mi_fet0.9",  "mi_fet0.8",  "mi_fet0.7",
  "mi_fet0.6",  "mi_fet0.5","mi_fet0.4",  "mi_fet0.3",  "mi_fet0.2",
  "mi_fet0.1",  "mi_fet0.08","mi_fet0.06","mi_fet0.04", "mi_fet0.02","mi_fet0.01",
  "tx_fetall", "tx_fet1", "tx_fet0.9", "tx_fet0.8", "tx_fet0.7",
  "tx_fet0.6", "tx_fet0.5","tx_fet0.4","tx_fet0.3", "tx_fet0.2",
  "tx_fet0.1", "tx_fet0.08","tx_fet0.06","tx_fet0.04","tx_fet0.02","tx_fet0.01",
  "DIFF_fetall", "DIFF_fet1", "DIFF_fet0.9", "DIFF_fet0.8", "DIFF_fet0.7",
  "DIFF_fet0.6", "DIFF_fet0.5","DIFF_fet0.4","DIFF_fet0.3", "DIFF_fet0.2",
  "DIFF_fet0.1", "DIFF_fet0.08","DIFF_fet0.06","DIFF_fet0.04","DIFF_fet0.02","DIFF_fet0.01")

# ---- T model: mi (all features) --------------------------------------------
ETA <- list(pc = list(X = data_tpm_mi[rownames(pheno_mi_trn), ], model = 'BRR'))

fm_brr <- BGLR(y      = Y_trn_mi[, trait],
               ETA    = ETA,
               nIter  = nIter,
               burnIn = burnIn)

save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_mi)[trait], "BRR_mi_TWgs_data", dataset, "_all.RData"))

# Extract and save marker effect estimates
b_tpm   <- as.matrix(fm_brr$ETA$pc$b)
write.csv(b_tpm, paste0("models/BRR/feature_importance/feature_importance_", colnames(Y_trn_mi)[trait], "BRR_mi_TWgs_data", dataset, "featureimp_all.csv"))

mu_trn  <- fm_brr$mu  # Intercept

# Predict on test set (excluding NAs)
data_frames_tst <- data_tpm_mi[rownames(pheno_mi_tst), ]
index   <- which(!is.na(Y_tst_mi[, trait]))
tst_pred <- data_frames_tst[index, ]
y_tst_n  <- Y_tst_mi[index, ]

pred_tst <- (as.matrix(tst_pred) %*% b_tpm) + mu_trn
colnames(pred_tst) <- "yHat"
accuracy_T[1, ] <- r2_cor(y_tst_n[, trait], pred_tst)

# ---- T model: mi (top % features from GB importance) -----------------------
for (i in 1:length(runs)) {

  # Select top fraction of features ranked by GB importance
  feature_mi_runs <- feature_mi[1:round(nrow(feature_mi) * runs[i]), ]

  ETA <- list(pc = list(X = data_tpm_mi[rownames(pheno_mi_trn), rownames(feature_mi_runs)], model = 'BRR'))

  fm_brr <- BGLR(y      = Y_trn_mi[, trait],
                 ETA    = ETA,
                 nIter  = nIter,
                 burnIn = burnIn)

  b_tpm  <- as.matrix(fm_brr$ETA$pc$b)
  write.csv(b_tpm, paste0("models/BRR/feature_importance/feature_importance_", colnames(Y_trn_mi)[trait], "BRR_mi_TWgs_data", dataset, "_featureimp_", runs[i], ".csv"))

  mu_trn <- fm_brr$mu

  data_frames_tst <- data_tpm_mi[rownames(pheno_mi_tst), rownames(feature_mi_runs)]
  index   <- which(!is.na(Y_tst_mi[, trait]))
  tst_pred <- data_frames_tst[index, ]
  y_tst_n  <- Y_tst_mi[index, ]

  pred_tst <- (as.matrix(tst_pred) %*% b_tpm) + mu_trn
  colnames(pred_tst) <- "yHat"
  accuracy_T[i + 1, ] <- r2_cor(y_tst_n[, trait], pred_tst)
}

# ---- T model: tx (all features) -------------------------------------------
ETA <- list(pc = list(X = data_tpm_tx[rownames(pheno_tx_trn), ], model = 'BRR'))

fm_brr <- BGLR(y      = Y_trn_tx[, trait],
               ETA    = ETA,
               nIter  = nIter,
               burnIn = burnIn)

save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_tx)[trait], "BRR_tx_TWgs_data", dataset, "_all.RData"))

b_tpm  <- as.matrix(fm_brr$ETA$pc$b)
write.csv(b_tpm, paste0("models/BRR/feature_importance/feature_importance_", colnames(Y_trn_tx)[trait], "BRR_tx_TWgs_data", dataset, "featureimp_all.csv"))

mu_trn <- fm_brr$mu

data_frames_tst <- data_tpm_tx[rownames(pheno_tx_tst), ]
index   <- which(!is.na(Y_tst_tx[, trait]))
tst_pred <- data_frames_tst[index, ]
y_tst_n  <- Y_tst_tx[index, ]

pred_tst <- (as.matrix(tst_pred) %*% b_tpm) + mu_trn
colnames(pred_tst) <- "yHat"
accuracy_T[17, ] <- r2_cor(y_tst_n[, trait], pred_tst)

# ---- T model: tx (top % features) -----------------------------------------
for (i in 1:length(runs)) {

  feature_tx_runs <- feature_tx[1:round(nrow(feature_tx) * runs[i]), ]
  ETA <- list(pc = list(X = data_tpm_tx[rownames(pheno_tx_trn), rownames(feature_tx_runs)], model = 'BRR'))

  fm_brr <- BGLR(y      = Y_trn_tx[, trait],
                 ETA    = ETA,
                 nIter  = nIter,
                 burnIn = burnIn)

  save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_tx)[trait], "BRR_tx_TWgs_data", dataset, "_perc", runs[i], ".RData"))

  b_tpm  <- as.matrix(fm_brr$ETA$pc$b)
  write.csv(b_tpm, paste0("models/BRR/feature_importance/feature_importance_", colnames(Y_trn_tx)[trait], "BRR_tx_TWgs_data", dataset, "_featureimp_", runs[i], ".csv"))

  mu_trn <- fm_brr$mu

  data_frames_tst <- data_tpm_tx[rownames(pheno_tx_tst), rownames(feature_tx_runs)]
  index   <- which(!is.na(Y_tst_tx[, trait]))
  tst_pred <- data_frames_tst[index, ]
  y_tst_n  <- Y_tst_tx[index, ]

  pred_tst <- (as.matrix(tst_pred) %*% b_tpm) + mu_trn
  colnames(pred_tst) <- "yHat"
  accuracy_T[i + 17, ] <- r2_cor(y_tst_n[, trait], pred_tst)
}

# ---- T model: DIFF/plasticity (all features) --------------------------------
ETA <- list(pc = list(X = data_tpm_diff[rownames(pheno_diff_trn), ], model = 'BRR'))

fm_brr <- BGLR(y      = Y_trn_diff[, trait],
               ETA    = ETA,
               nIter  = nIter,
               burnIn = burnIn)

save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_diff)[trait], "BRR_DIFF_TWgs_data", dataset, "_all.RData"))

b_tpm  <- as.matrix(fm_brr$ETA$pc$b)
write.csv(b_tpm, paste0("models/BRR/feature_importance/feature_importance_", colnames(Y_trn_diff)[trait], "BRR_DIFF_TWgs_data", dataset, "_featureimp_all.csv"))

mu_trn <- fm_brr$mu

data_frames_tst <- data_tpm_diff[rownames(pheno_diff_tst), ]
index   <- which(!is.na(Y_tst_diff[, trait]))
tst_pred <- data_frames_tst[index, ]
y_tst_n  <- Y_tst_diff[index, ]
pred_tst <- (as.matrix(tst_pred) %*% b_tpm) + mu_trn
colnames(pred_tst) <- "yHat"
accuracy_T[33, ] <- r2_cor(y_tst_n[, trait], pred_tst)

# ---- T model: DIFF/plasticity (top % features) ------------------------------
for (i in 1:length(runs)) {

  feature_diff_runs <- feature_diff[1:round(nrow(feature_diff) * runs[i]), ]
  ETA <- list(pc = list(X = data_tpm_diff[rownames(pheno_diff_trn), rownames(feature_diff_runs)], model = 'BRR'))

  fm_brr <- BGLR(y      = Y_trn_diff[, trait],
                 ETA    = ETA,
                 nIter  = nIter,
                 burnIn = burnIn)

  save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_diff)[trait], "BRR_DIFF_TWgs_data", dataset, "_perc", runs[i], ".RData"))

  b_tpm <- as.matrix(fm_brr$ETA$pc$b)
  write.csv(b_tpm, paste0("models/BRR/feature_importance/feature_importance_", colnames(Y_trn_diff)[trait], "BRR_DIFF_TWgs_data", dataset, "_featureimp_", runs[i], ".csv"))

  mu_trn <- fm_brr$mu

  data_frames_tst <- data_tpm_diff[rownames(pheno_diff_tst), rownames(feature_diff_runs)]
  index   <- which(!is.na(Y_tst_diff[, trait]))
  tst_pred <- data_frames_tst[index, ]
  y_tst_n  <- Y_tst_diff[index, ]

  pred_tst <- (as.matrix(tst_pred) %*% b_tpm) + mu_trn
  colnames(pred_tst) <- "yHat"
  accuracy_T[i + 33, ] <- r2_cor(y_tst_n[, trait], pred_tst)
}

# Save T results
write.csv(accuracy_T, paste0("models/BRR/results/", colnames(Y_trn_diff)[trait], "_T_BRR.csv"))

# =============================================================================
# SECTION 2: SNP (G) BRR models
# =============================================================================

accuracy_G <- matrix(0, nrow = 48, ncol = 2)
colnames(accuracy_G) <- c("r2", "pCor")
rownames(accuracy_G) <- c(
  "mi_fetall",  "mi_fet1",  "mi_fet0.9",  "mi_fet0.8",  "mi_fet0.7",
  "mi_fet0.6",  "mi_fet0.5","mi_fet0.4",  "mi_fet0.3",  "mi_fet0.2",
  "mi_fet0.1",  "mi_fet0.08","mi_fet0.06","mi_fet0.04", "mi_fet0.02","mi_fet0.01",
  "tx_fetall", "tx_fet1", "tx_fet0.9", "tx_fet0.8", "tx_fet0.7",
  "tx_fet0.6", "tx_fet0.5","tx_fet0.4","tx_fet0.3", "tx_fet0.2",
  "tx_fet0.1", "tx_fet0.08","tx_fet0.06","tx_fet0.04","tx_fet0.02","tx_fet0.01",
  "DIFF_fetall", "DIFF_fet1", "DIFF_fet0.9", "DIFF_fet0.8", "DIFF_fet0.7",
  "DIFF_fet0.6", "DIFF_fet0.5","DIFF_fet0.4","DIFF_fet0.3", "DIFF_fet0.2",
  "DIFF_fet0.1", "DIFF_fet0.08","DIFF_fet0.06","DIFF_fet0.04","DIFF_fet0.02","DIFF_fet0.01")

# ---- G model: mi (all features) --------------------------------------------
ETA <- list(pc = list(X = snps[rownames(Y_trn_mi), ], model = 'BRR'))

fm_brr <- BGLR(y      = Y_trn_mi[, trait],
               ETA    = ETA,
               nIter  = nIter,
               burnIn = burnIn)

save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_mi)[trait], "BRR_mi_G_data", dataset, "_all.RData"))

b      <- as.matrix(fm_brr$ETA$pc$b)
write.csv(b, paste0("models/BRR/feature_importance/feature_importance_", colnames(Y_trn_mi)[trait], "BRR_mi_G_data", dataset, "featureimp_all.csv"))
mu_trn <- fm_brr$mu

data_frames_tst <- snps[rownames(pheno_mi_tst), ]
index   <- which(!is.na(Y_tst_mi[, trait]))
tst_pred <- data_frames_tst[index, ]
y_tst_n  <- Y_tst_mi[index, ]

pred_tst <- (as.matrix(tst_pred) %*% b) + mu_trn
colnames(pred_tst) <- "yHat"
accuracy_G[1, ] <- r2_cor(y_tst_n[, trait], pred_tst)

# ---- G model: mi (top % features) ------------------------------------------
for (i in 1:length(runs)) {

  feature_mi_runs <- feature_mi_G[1:round(nrow(feature_mi_G) * runs[i]), ]
  ETA <- list(pc = list(X = snps[rownames(Y_trn_mi), rownames(feature_mi_runs)], model = 'BRR'))

  fm_brr <- BGLR(y      = Y_trn_mi[, trait],
                 ETA    = ETA,
                 nIter  = nIter,
                 burnIn = burnIn)

  save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_mi)[trait], "BRR_mi_G_data", dataset, "_perc", runs[i], ".RData"))

  b      <- as.matrix(fm_brr$ETA$pc$b)
  write.csv(b, paste0("models/BRR/feature_importance/feature_importance_", colnames(Y_trn_mi)[trait], "BRR_mi_G_data", dataset, "_featureimp_", runs[i], ".csv"))
  mu_trn <- fm_brr$mu

  data_frames_tst <- snps[rownames(pheno_mi_tst), rownames(feature_mi_runs)]
  index   <- which(!is.na(Y_tst_mi[, trait]))
  tst_pred <- data_frames_tst[index, ]
  y_tst_n  <- Y_tst_mi[index, ]

  pred_tst <- (as.matrix(tst_pred) %*% b) + mu_trn
  colnames(pred_tst) <- "yHat"
  accuracy_G[i + 1, ] <- r2_cor(y_tst_n[, trait], pred_tst)
}

# ---- G model: tx (all features) -------------------------------------------
ETA <- list(pc = list(X = snps[rownames(pheno_tx_trn), ], model = 'BRR'))

fm_brr <- BGLR(y      = Y_trn_tx[, trait],
               ETA    = ETA,
               nIter  = nIter,
               burnIn = burnIn)

save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_tx)[trait], "BRR_tx_G_data", dataset, "_all.RData"))

b      <- as.matrix(fm_brr$ETA$pc$b)
write.csv(b, paste0("models/BRR/feature_importance/feature_importance_", colnames(Y_trn_tx)[trait], "BRR_tx_T_data", dataset, "featureimp_all.csv"))
mu_trn <- fm_brr$mu

data_frames_tst <- snps[rownames(pheno_tx_tst), ]
index   <- which(!is.na(Y_tst_tx[, trait]))
tst_pred <- data_frames_tst[index, ]
y_tst_n  <- Y_tst_tx[index, ]

pred_tst <- (as.matrix(tst_pred) %*% b) + mu_trn
colnames(pred_tst) <- "yHat"
accuracy_G[17, ] <- r2_cor(y_tst_n[, trait], pred_tst)

# ---- G model: tx (top % features) -----------------------------------------
for (i in 1:length(runs)) {

  feature_tx_runs <- feature_tx_G[1:round(nrow(feature_tx_G) * runs[i]), ]
  ETA <- list(pc = list(X = snps[rownames(pheno_tx_trn), rownames(feature_tx_runs)], model = 'BRR'))

  fm_brr <- BGLR(y      = Y_trn_tx[, trait],
                 ETA    = ETA,
                 nIter  = nIter,
                 burnIn = burnIn)

  save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_tx)[trait], "BRR_tx_G_data", dataset, "_perc", runs[i], ".RData"))

  b      <- as.matrix(fm_brr$ETA$pc$b)
  write.csv(b, paste0("models/BRR/feature_importance/feature_importance_", colnames(Y_trn_tx)[trait], "BRR_tx_G_data", dataset, "_featureimp_", runs[i], ".csv"))
  mu_trn <- fm_brr$mu

  data_frames_tst <- snps[rownames(pheno_tx_tst), rownames(feature_tx_runs)]
  index   <- which(!is.na(Y_tst_tx[, trait]))
  tst_pred <- data_frames_tst[index, ]
  y_tst_n  <- Y_tst_tx[index, ]

  pred_tst <- (as.matrix(tst_pred) %*% b) + mu_trn
  colnames(pred_tst) <- "yHat"
  accuracy_G[i + 17, ] <- r2_cor(y_tst_n[, trait], pred_tst)
}

# ---- G model: DIFF/plasticity (all features) --------------------------------
ETA <- list(pc = list(X = snps[rownames(pheno_diff_trn), ], model = 'BRR'))

fm_brr <- BGLR(y      = Y_trn_diff[, trait],
               ETA    = ETA,
               nIter  = nIter,
               burnIn = burnIn)

save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_diff)[trait], "BRR_DIFF_G_data", dataset, "_all.RData"))

b      <- as.matrix(fm_brr$ETA$pc$b)
write.csv(b, paste0("models/BRR/feature_importance/feature_importance_", colnames(Y_trn_diff)[trait], "BRR_DIFF_G_data", dataset, "_featureimp_all.csv"))
mu_trn <- fm_brr$mu

data_frames_tst <- snps[rownames(pheno_diff_tst), ]
index   <- which(!is.na(Y_tst_diff[, trait]))
tst_pred <- data_frames_tst[index, ]
y_tst_n  <- Y_tst_diff[index, ]
pred_tst <- (as.matrix(tst_pred) %*% b) + mu_trn
colnames(pred_tst) <- "yHat"
accuracy_G[33, ] <- r2_cor(y_tst_n[, trait], pred_tst)

# ---- G model: DIFF/plasticity (top % features) ------------------------------
for (i in 1:length(runs)) {

  feature_diff_runs <- feature_diff_G[1:round(nrow(feature_diff_G) * runs[i]), ]
  ETA <- list(pc = list(X = snps[rownames(pheno_diff_trn), rownames(feature_diff_runs)], model = 'BRR'))

  fm_brr <- BGLR(y      = Y_trn_diff[, trait],
                 ETA    = ETA,
                 nIter  = nIter,
                 burnIn = burnIn)

  save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_diff)[trait], "BRR_DIFF_G_data", dataset, "_perc", runs[i], ".RData"))

  b      <- as.matrix(fm_brr$ETA$pc$b)
  write.csv(b, paste0("models/BRR/feature_importance/feature_importance_", colnames(Y_trn_diff)[trait], "BRR_DIFF_G_data", dataset, "_featureimp_", runs[i], ".csv"))
  mu_trn <- fm_brr$mu

  data_frames_tst <- snps[rownames(pheno_diff_tst), rownames(feature_diff_runs)]
  index   <- which(!is.na(Y_tst_diff[, trait]))
  tst_pred <- data_frames_tst[index, ]
  y_tst_n  <- Y_tst_diff[index, ]

  pred_tst <- (as.matrix(tst_pred) %*% b) + mu_trn
  colnames(pred_tst) <- "yHat"
  accuracy_G[i + 33, ] <- r2_cor(y_tst_n[, trait], pred_tst)
}

# =============================================================================
# SECTION 3: Multi-omic (GT: SNP + TPM) BRR models
# =============================================================================

accuracy_GT <- matrix(0, nrow = 48, ncol = 2)
colnames(accuracy_GT) <- c("r2", "pCor")
rownames(accuracy_GT) <- c(
  "mi_fetall",  "mi_fet1",  "mi_fet0.9",  "mi_fet0.8",  "mi_fet0.7",
  "mi_fet0.6",  "mi_fet0.5","mi_fet0.4",  "mi_fet0.3",  "mi_fet0.2",
  "mi_fet0.1",  "mi_fet0.08","mi_fet0.06","mi_fet0.04", "mi_fet0.02","mi_fet0.01",
  "tx_fetall", "tx_fet1", "tx_fet0.9", "tx_fet0.8", "tx_fet0.7",
  "tx_fet0.6", "tx_fet0.5","tx_fet0.4","tx_fet0.3", "tx_fet0.2",
  "tx_fet0.1", "tx_fet0.08","tx_fet0.06","tx_fet0.04","tx_fet0.02","tx_fet0.01",
  "DIFF_fetall", "DIFF_fet1", "DIFF_fet0.9", "DIFF_fet0.8", "DIFF_fet0.7",
  "DIFF_fet0.6", "DIFF_fet0.5","DIFF_fet0.4","DIFF_fet0.3", "DIFF_fet0.2",
  "DIFF_fet0.1", "DIFF_fet0.08","DIFF_fet0.06","DIFF_fet0.04","DIFF_fet0.02","DIFF_fet0.01")

# ---- GT model: mi (all features) -------------------------------------------
ETA <- list(pc = list(X = data_tpm_mi_snps[rownames(Y_trn_mi), ], model = 'BRR'))

fm_brr <- BGLR(y      = Y_trn_mi[, trait],
               ETA    = ETA,
               nIter  = nIter,
               burnIn = burnIn)

save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_mi)[trait], "BRR_mi_GT_data", dataset, "_all.RData"))

b      <- as.matrix(fm_brr$ETA$pc$b)
write.csv(b, paste0("models/BRR/feature_importance/feature_importance_", colnames(Y_trn_mi)[trait], "BRR_mi_GT_data", dataset, "featureimp_all.csv"))
mu_trn <- fm_brr$mu

data_frames_tst <- data_tpm_mi_snps[rownames(pheno_mi_tst), ]
index   <- which(!is.na(Y_tst_mi[, trait]))
tst_pred <- data_frames_tst[index, ]
y_tst_n  <- Y_tst_mi[index, ]

pred_tst <- (as.matrix(tst_pred) %*% b) + mu_trn
colnames(pred_tst) <- "yHat"
accuracy_GT[1, ] <- r2_cor(y_tst_n[, trait], pred_tst)

# ---- GT model: mi (top % features) -----------------------------------------
for (i in 1:length(runs)) {

  feature_mi_runs <- feature_mi_GT[1:round(nrow(feature_mi_GT) * runs[i]), ]
  ETA <- list(pc = list(X = data_tpm_mi_snps[rownames(Y_trn_mi), rownames(feature_mi_runs)], model = 'BRR'))

  fm_brr <- BGLR(y      = Y_trn_mi[, trait],
                 ETA    = ETA,
                 nIter  = nIter,
                 burnIn = burnIn)

  save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_mi)[trait], "BRR_mi_GT_data", dataset, "_perc", runs[i], ".RData"))

  b      <- as.matrix(fm_brr$ETA$pc$b)
  write.csv(b, paste0("models/BRR/feature_importance/feature_importance_", colnames(Y_trn_mi)[trait], "BRR_mi_GT_data", dataset, "_featureimp_", runs[i], ".csv"))
  mu_trn <- fm_brr$mu

  data_frames_tst <- data_tpm_mi_snps[rownames(pheno_mi_tst), rownames(feature_mi_runs)]
  index   <- which(!is.na(Y_tst_mi[, trait]))
  tst_pred <- data_frames_tst[index, ]
  y_tst_n  <- Y_tst_mi[index, ]

  pred_tst <- (as.matrix(tst_pred) %*% b) + mu_trn
  colnames(pred_tst) <- "yHat"
  accuracy_GT[i + 1, ] <- r2_cor(y_tst_n[, trait], pred_tst)
}

# ---- GT model: tx (all features) ------------------------------------------
ETA <- list(pc = list(X = data_tpm_tx_snps[rownames(pheno_tx_trn), ], model = 'BRR'))

fm_brr <- BGLR(y      = Y_trn_tx[, trait],
               ETA    = ETA,
               nIter  = nIter,
               burnIn = burnIn)

save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_tx)[trait], "BRR_tx_GT_data", dataset, "_all.RData"))

b      <- as.matrix(fm_brr$ETA$pc$b)
write.csv(b, paste0("models/BRR/feature_importance/feature_importance_", colnames(Y_trn_tx)[trait], "BRR_tx_GT_data", dataset, "featureimp_all.csv"))
mu_trn <- fm_brr$mu

data_frames_tst <- data_tpm_tx_snps[rownames(pheno_tx_tst), ]
index   <- which(!is.na(Y_tst_tx[, trait]))
tst_pred <- data_frames_tst[index, ]
y_tst_n  <- Y_tst_tx[index, ]

pred_tst <- (as.matrix(tst_pred) %*% b) + mu_trn
colnames(pred_tst) <- "yHat"
accuracy_GT[17, ] <- r2_cor(y_tst_n[, trait], pred_tst)

# ---- GT model: tx (top % features) ----------------------------------------
for (i in 1:length(runs)) {

  feature_tx_runs <- feature_tx_GT[1:round(nrow(feature_tx_GT) * runs[i]), ]
  ETA <- list(pc = list(X = data_tpm_tx_snps[rownames(pheno_tx_trn), rownames(feature_tx_runs)], model = 'BRR'))

  fm_brr <- BGLR(y      = Y_trn_tx[, trait],
                 ETA    = ETA,
                 nIter  = nIter,
                 burnIn = burnIn)

  save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_tx)[trait], "BRR_tx_GT_data", dataset, "_perc", runs[i], ".RData"))

  b      <- as.matrix(fm_brr$ETA$pc$b)
  write.csv(b, paste0("models/BRR/feature_importance/feature_importance_", colnames(Y_trn_tx)[trait], "BRR_tx_GT_data", dataset, "_featureimp_", runs[i], ".csv"))
  mu_trn <- fm_brr$mu

  data_frames_tst <- data_tpm_tx_snps[rownames(pheno_tx_tst), rownames(feature_tx_runs)]
  index   <- which(!is.na(Y_tst_tx[, trait]))
  tst_pred <- data_frames_tst[index, ]
  y_tst_n  <- Y_tst_tx[index, ]

  pred_tst <- (as.matrix(tst_pred) %*% b) + mu_trn
  colnames(pred_tst) <- "yHat"
  accuracy_GT[i + 17, ] <- r2_cor(y_tst_n[, trait], pred_tst)
}

# ---- GT model: DIFF/plasticity (all features) -------------------------------
ETA <- list(pc = list(X = data_tpm_diff_snps[rownames(pheno_diff_trn), ], model = 'BRR'))

fm_brr <- BGLR(y      = Y_trn_diff[, trait],
               ETA    = ETA,
               nIter  = nIter,
               burnIn = burnIn)

save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_diff)[trait], "BRR_DIFF_GT_data", dataset, "_all.RData"))

b      <- as.matrix(fm_brr$ETA$pc$b)
write.csv(b, paste0("models/BRR/feature_importance/feature_importance_", colnames(Y_trn_diff)[trait], "BRR_DIFF_GT_data", dataset, "_featureimp_all.csv"))
mu_trn <- fm_brr$mu

data_frames_tst <- data_tpm_diff_snps[rownames(pheno_diff_tst), ]
index   <- which(!is.na(Y_tst_diff[, trait]))
tst_pred <- data_frames_tst[index, ]
y_tst_n  <- Y_tst_diff[index, ]
pred_tst <- (as.matrix(tst_pred) %*% b) + mu_trn
colnames(pred_tst) <- "yHat"
accuracy_GT[33, ] <- r2_cor(y_tst_n[, trait], pred_tst)

# ---- GT model: DIFF/plasticity (top % features) -----------------------------
for (i in 1:length(runs)) {

  feature_diff_runs <- feature_diff_GT[1:round(nrow(feature_diff_GT) * runs[i]), ]
  ETA <- list(pc = list(X = data_tpm_diff_snps[rownames(pheno_diff_trn), rownames(feature_diff_runs)], model = 'BRR'))

  fm_brr <- BGLR(y      = Y_trn_diff[, trait],
                 ETA    = ETA,
                 nIter  = nIter,
                 burnIn = burnIn)

  save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_diff)[trait], "BRR_DIFF_GT_data", dataset, "_perc", runs[i], ".RData"))

  b      <- as.matrix(fm_brr$ETA$pc$b)
  write.csv(b, paste0("models/BRR/feature_importance/feature_importance_", colnames(Y_trn_diff)[trait], "BRR_DIFF_GT_data", dataset, "_featureimp_", runs[i], ".csv"))
  mu_trn <- fm_brr$mu

  data_frames_tst <- data_tpm_diff_snps[rownames(pheno_diff_tst), rownames(feature_diff_runs)]
  index   <- which(!is.na(Y_tst_diff[, trait]))
  tst_pred <- data_frames_tst[index, ]
  y_tst_n  <- Y_tst_diff[index, ]

  pred_tst <- (as.matrix(tst_pred) %*% b) + mu_trn
  colnames(pred_tst) <- "yHat"
  accuracy_GT[i + 33, ] <- r2_cor(y_tst_n[, trait], pred_tst)
}

# =============================================================================
# SECTION 4: PCA-based BRR models (first 5 PCs as input)
# Used as a dimensionality-reduction baseline for comparison
# =============================================================================

# Results matrix: 9 rows = 3 feature sets (T, G, GT) x 3 environments (mi, tx, DIFF)
accuracy_PC <- matrix(0, nrow = 9, ncol = 2)
colnames(accuracy_PC) <- c("r2", "pCor")
rownames(accuracy_PC)  <- c("mi_T",  "tx_T",  "DIFF_T",
                             "mi_G",  "tx_G",  "DIFF_G",
                             "mi_GT", "tx_GT", "DIFF_GT")

# ---- PCA-T: mi -------------------------------------------------------------
pca_mi_T     <- prcomp(data_tpm_mi[rownames(pheno_mi_trn), ])
pca_mi_T_trn <- data.frame(pca_mi_T$x)
pca_mi_T_tst <- predict(pca_mi_T, data_tpm_mi[rownames(pheno_mi_tst), ])  # Project test onto training PCs

ETA <- list(pc = list(X = pca_mi_T_trn[, 1:5], model = 'BRR'))
fm_brr <- BGLR(y = Y_trn_mi[, trait], ETA = ETA, nIter = nIter, burnIn = burnIn)
save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_mi)[trait], "BRR_mi_T_PC_data", dataset, ".RData"))

b      <- as.matrix(fm_brr$ETA$pc$b)
mu_trn <- fm_brr$mu
index  <- which(!is.na(Y_tst_mi[, trait]))
pred_tst <- (as.matrix(pca_mi_T_tst[index, 1:5]) %*% b) + mu_trn
colnames(pred_tst) <- "yHat"
accuracy_PC[1, ] <- r2_cor(Y_tst_mi[index, trait], pred_tst)

# ---- PCA-T: tx ------------------------------------------------------------
pca_tx_T     <- prcomp(data_tpm_tx[rownames(pheno_tx_trn), ])
pca_tx_T_trn <- data.frame(pca_tx_T$x)
pca_tx_T_tst <- predict(pca_tx_T, data_tpm_tx[rownames(pheno_tx_tst), ])

ETA <- list(pc = list(X = pca_tx_T_trn[, 1:5], model = 'BRR'))
fm_brr <- BGLR(y = Y_trn_tx[, trait], ETA = ETA, nIter = nIter, burnIn = burnIn)
save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_mi)[trait], "BRR_tx_T_PC_data", dataset, ".RData"))

b      <- as.matrix(fm_brr$ETA$pc$b)
mu_trn <- fm_brr$mu
index  <- which(!is.na(Y_tst_tx[, trait]))
pred_tst <- (as.matrix(pca_tx_T_tst[index, 1:5]) %*% b) + mu_trn
colnames(pred_tst) <- "yHat"
accuracy_PC[2, ] <- r2_cor(Y_tst_tx[index, trait], pred_tst)

# ---- PCA-T: DIFF ------------------------------------------------------------
pca_diff_T     <- prcomp(data_tpm_diff[rownames(pheno_diff_trn), ])
pca_diff_T_trn <- data.frame(pca_diff_T$x)
pca_diff_T_tst <- predict(pca_diff_T, data_tpm_diff[rownames(pheno_diff_tst), ])

ETA <- list(pc = list(X = pca_diff_T_trn[, 1:5], model = 'BRR'))
fm_brr <- BGLR(y = Y_trn_diff[, trait], ETA = ETA, nIter = nIter, burnIn = burnIn)
save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_diff)[trait], "BRR_DIFF_T_PC_data", dataset, ".RData"))

b      <- as.matrix(fm_brr$ETA$pc$b)
mu_trn <- fm_brr$mu
index  <- which(!is.na(Y_tst_diff[, trait]))
pred_tst <- (as.matrix(pca_diff_T_tst[index, 1:5]) %*% b) + mu_trn
colnames(pred_tst) <- "yHat"
accuracy_PC[3, ] <- r2_cor(Y_tst_diff[index, trait], pred_tst)

# ---- PCA-G: mi -------------------------------------------------------------
pca_mi_G     <- prcomp(snps[rownames(Y_trn_mi), ])
pca_mi_G_trn <- data.frame(pca_mi_G$x)
pca_mi_G_tst <- predict(pca_mi_G, snps[rownames(pheno_mi_tst), ])

ETA <- list(pc = list(X = pca_mi_G_trn[, 1:5], model = 'BRR'))
fm_brr <- BGLR(y = Y_trn_mi[, trait], ETA = ETA, nIter = nIter, burnIn = burnIn)
save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_mi)[trait], "BRR_mi_G_PC_data", dataset, ".RData"))

b      <- as.matrix(fm_brr$ETA$pc$b)
mu_trn <- fm_brr$mu
index  <- which(!is.na(Y_tst_mi[, trait]))
pred_tst <- (as.matrix(pca_mi_G_tst[index, 1:5]) %*% b) + mu_trn
colnames(pred_tst) <- "yHat"
accuracy_PC[4, ] <- r2_cor(Y_tst_mi[index, trait], pred_tst)

# ---- PCA-G: tx ------------------------------------------------------------
pca_tx_G     <- prcomp(snps[rownames(Y_trn_tx), ])
pca_tx_G_trn <- data.frame(pca_tx_G$x)
pca_tx_G_tst <- predict(pca_tx_G, snps[rownames(pheno_tx_tst), ])

ETA <- list(pc = list(X = pca_tx_G_trn[, 1:5], model = 'BRR'))
fm_brr <- BGLR(y = Y_trn_tx[, trait], ETA = ETA, nIter = nIter, burnIn = burnIn)
save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_mi)[trait], "BRR_tx_G_PC_data", dataset, ".RData"))

b      <- as.matrix(fm_brr$ETA$pc$b)
mu_trn <- fm_brr$mu
index  <- which(!is.na(Y_tst_tx[, trait]))
pred_tst <- (as.matrix(pca_tx_G_tst[index, 1:5]) %*% b) + mu_trn
colnames(pred_tst) <- "yHat"
accuracy_PC[5, ] <- r2_cor(Y_tst_tx[index, trait], pred_tst)

# ---- PCA-G: DIFF ------------------------------------------------------------
pca_diff_G     <- prcomp(snps[rownames(Y_trn_diff), ])
pca_diff_G_trn <- data.frame(pca_diff_G$x)
pca_diff_G_tst <- predict(pca_diff_G, snps[rownames(pheno_diff_tst), ])

ETA <- list(pc = list(X = pca_diff_G_trn[, 1:5], model = 'BRR'))
fm_brr <- BGLR(y = Y_trn_diff[, trait], ETA = ETA, nIter = nIter, burnIn = burnIn)
save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_diff)[trait], "BRR_DIFF_G_PC_data", dataset, ".RData"))

b      <- as.matrix(fm_brr$ETA$pc$b)
mu_trn <- fm_brr$mu
index  <- which(!is.na(Y_tst_diff[, trait]))
pred_tst <- (as.matrix(pca_diff_G_tst[index, 1:5]) %*% b) + mu_trn
colnames(pred_tst) <- "yHat"
accuracy_PC[6, ] <- r2_cor(Y_tst_diff[index, trait], pred_tst)

# ---- PCA-GT: mi ------------------------------------------------------------
pca_mi_GT     <- prcomp(data_tpm_mi_snps[rownames(Y_trn_mi), ])
pca_mi_GT_trn <- data.frame(pca_mi_GT$x)
pca_mi_GT_tst <- predict(pca_mi_GT, data_tpm_mi_snps[rownames(pheno_mi_tst), ])

ETA <- list(pc = list(X = pca_mi_GT_trn[, 1:5], model = 'BRR'))
fm_brr <- BGLR(y = Y_trn_mi[, trait], ETA = ETA, nIter = nIter, burnIn = burnIn)
save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_mi)[trait], "BRR_mi_GT_PC_data", dataset, ".RData"))

b      <- as.matrix(fm_brr$ETA$pc$b)
mu_trn <- fm_brr$mu
index  <- which(!is.na(Y_tst_mi[, trait]))
pred_tst <- (as.matrix(pca_mi_GT_tst[index, 1:5]) %*% b) + mu_trn
colnames(pred_tst) <- "yHat"
accuracy_PC[7, ] <- r2_cor(Y_tst_mi[index, trait], pred_tst)

# ---- PCA-GT: tx -----------------------------------------------------------
pca_tx_GT     <- prcomp(data_tpm_tx_snps[rownames(Y_trn_tx), ])
pca_tx_GT_trn <- data.frame(pca_tx_GT$x)
pca_tx_GT_tst <- predict(pca_tx_GT, data_tpm_tx_snps[rownames(pheno_tx_tst), ])

ETA <- list(pc = list(X = pca_tx_GT_trn[, 1:5], model = 'BRR'))
fm_brr <- BGLR(y = Y_trn_tx[, trait], ETA = ETA, nIter = nIter, burnIn = burnIn)
save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_mi)[trait], "BRR_tx_GT_PC_data", dataset, ".RData"))

b      <- as.matrix(fm_brr$ETA$pc$b)
mu_trn <- fm_brr$mu
index  <- which(!is.na(Y_tst_tx[, trait]))
pred_tst <- (as.matrix(pca_tx_GT_tst[index, 1:5]) %*% b) + mu_trn
colnames(pred_tst) <- "yHat"
accuracy_PC[8, ] <- r2_cor(Y_tst_tx[index, trait], pred_tst)

# ---- PCA-GT: DIFF -----------------------------------------------------------
pca_diff_GT     <- prcomp(data_tpm_diff_snps[rownames(Y_trn_diff), ])
pca_diff_GT_trn <- data.frame(pca_diff_GT$x)
pca_diff_GT_tst <- predict(pca_diff_GT, data_tpm_diff_snps[rownames(pheno_diff_tst), ])

ETA <- list(pc = list(X = pca_diff_GT_trn[, 1:5], model = 'BRR'))
fm_brr <- BGLR(y = Y_trn_diff[, trait], ETA = ETA, nIter = nIter, burnIn = burnIn)
save(fm_brr, file = paste0("models/BRR/", colnames(Y_trn_diff)[trait], "BRR_DIFF_GT_PC_data", dataset, ".RData"))

b      <- as.matrix(fm_brr$ETA$pc$b)
mu_trn <- fm_brr$mu
index  <- which(!is.na(Y_tst_diff[, trait]))
pred_tst <- (as.matrix(pca_diff_GT_tst[index, 1:5]) %*% b) + mu_trn
colnames(pred_tst) <- "yHat"
accuracy_PC[9, ] <- r2_cor(Y_tst_diff[index, trait], pred_tst)

# =============================================================================
# SECTION 5: Combine and save all results
# =============================================================================

# Add dataset label and feature percentage identifier to each results table
accuracy_G  <- as.data.frame(accuracy_G)  %>% mutate(dataset = "G",  perc = rownames(accuracy_G))  %>% remove_rownames()
accuracy_T  <- as.data.frame(accuracy_T)  %>% mutate(dataset = "T",  perc = rownames(accuracy_T))  %>% remove_rownames()
accuracy_GT <- as.data.frame(accuracy_GT) %>% mutate(dataset = "GT", perc = rownames(accuracy_GT)) %>% remove_rownames()
accuracy_PC <- as.data.frame(accuracy_PC) %>% mutate(dataset = "PC", perc = rownames(accuracy_PC)) %>% remove_rownames()

# Combine all model results into a single data frame
accuracy <- rbind(accuracy_PC, accuracy_T, accuracy_G, accuracy_GT)

# Save combined results for the current trait
write.csv(accuracy, paste0("models/BRR/results/", colnames(Y_trn_diff)[trait], "_G_T_GT_BRR_complete.csv"))