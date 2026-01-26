######################## Script: eCor, Kinship and Heritability ########################
# Description:
# This script performs a gene-by-environment (GxE) analysis for switchgrass RNA-seq data.
# It calculates expression correlation matrices (eCor), kinship matrix from SNPs,
# performs variance partitioning, and estimates heritability for each trait
# across environments (MI and TX) and their differences.
####################################################################################################

######################## Load Libraries ########################
rm(list = ls())  # Clear environment

# Load required packages
library(tidyverse)
library(lme4)
library(pheatmap)
library(lmerTest)
library(sommer)
library(data.table)

# Set working directory (adjust this as needed)
setwd("/Users/paulo/Library/CloudStorage/OneDrive-MichiganStateUniversity/Shiu_lab/switchgrass_rna_seq/common_samples/scripts_github")

######################## Load and Prepare Data ########################

# Load phenotype data from MI and TX environments
mi <- read.csv("../data/phenotype_MI.csv")
tx <- read.csv("../data/phenotype_TX.csv")
diff <- tx - mi  # Compute phenotype differences between environments
mi$env <- "MI"
tx$env <- "TX"

# Load RNA-seq expression data (TPM - log2 transformed)
tpm_mi <- fread("../data/rna_mi_ap13_nzv_log2.csv")
tpm_tx <- fread("../data/rna_tx_ap13_nzv_log2.csv")
tpm_diff <- tpm_tx - tpm_mi

# Load SNP data and format
snps <- fread("../data/WGS/snps.csv")
snps <- column_to_rownames(snps, var = "PLANT_ID")

# Convert all to matrix format for calculations
tpm_mi <- as.matrix(tpm_mi)
tpm_tx <- as.matrix(tpm_tx)
tpm_diff <- as.matrix(tpm_diff)
snps <- as.matrix(snps)

######################## Calculate Expression Correlation (eCor) Matrices ########################

# Calculate sample-to-sample correlation matrices for expression data
eCor_mi <- cor(t(tpm_mi))
eCor_tx <- cor(t(tpm_tx))
ecor_diff <- cor(t(tpm_diff))

# Save eCor matrices
write.csv(eCor_mi, "../data/eCor_mi.csv")
write.csv(eCor_tx, "../data/eCor_tx.csv")
write.csv(ecor_diff, "../data/eCor_diff.csv")

######################## Kinship Matrix Calculation ########################

# Standardize SNP matrix and calculate kinship using realized relationship matrix
M_snps <- scale(snps)
K <- tcrossprod(M_snps) / ncol(M_snps)

# Save kinship matrix
write.csv(K, "../data/kinship.csv")

######################## Variance Partitioning Using Mixed Models ########################

# Combine phenotype data across environments
pheno <- rbind(mi, tx)
colnames(pheno)[1] <- "genotype"  # Ensure consistent naming

# Function: Partition variance into G, E, and GxE components using sommer::mmer
variance_partition_mixed <- function(trait, data, kinship) {
  data$genotype <- as.factor(data$genotype)
  data$env <- as.factor(data$env)
  data$y <- data[[trait]]

  model <- mmer(
    y ~ 1,
    random = ~ env + vs(genotype, Gu = kinship) + genotype:env,
    data = data
  )

  var_comps <- summary(model)$varcomp
  total_var <- sum(var_comps$VarComp)
  percent_explained <- round(100 * var_comps$VarComp / total_var, 2)

  variance_summary <- data.frame(
    Component = rownames(var_comps),
    Variance = var_comps$VarComp,
    Percent = percent_explained
  )

  return(variance_summary)
}

# Apply to each trait and save outputs
traits <- c("GR", "EM", "FL", "TC", "HT", "logBiomass")
for (trait in traits) {
  vp <- variance_partition_mixed(trait, pheno, K)
  write.csv(vp, paste0("data/variance_partition_mixed_", trait, ".csv"), row.names = FALSE)
}

######################## Heritability Estimation ########################

# Create results dataframe
h2_traits <- data.frame(MI = rep(NA, 6), TX = rep(NA, 6), Diff = rep(NA, 6))
rownames(h2_traits) <- colnames(mi)[2:7]  # Assuming traits are columns 2-7

# Function: Estimate narrow-sense heritability using BGLR
calculate_heritability <- function(y, K) {
  fm <- BGLR(y = y, ETA = list(list(X = K, model = 'BRR', saveEffects = TRUE)),
             nIter = 12000, burnIn = 2000, verbose = FALSE)
  varU <- scan('ETA_1_varB.dat')
  varE <- scan('varE.dat')
  h2 <- varU / (varU + varE)
  mean(h2)
}

# Estimate heritability for MI, TX, and the difference for each trait
for (i in 1:nrow(h2_traits)) {
  h2_traits$MI[i] <- calculate_heritability(mi[, i+1], K)
  h2_traits$TX[i] <- calculate_heritability(tx[, i+1], K)
  h2_traits$Diff[i] <- calculate_heritability(diff[, i], K)
}

# Save heritability estimates
write.csv(h2_traits, "data/heritability_traits.csv")

####################################################################################################
# End of Script