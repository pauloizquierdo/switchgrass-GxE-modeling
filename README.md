# 🧬 Predictive modeling using gene expression in Switchgrass

This repository contains code and scripts for the study:

**"Predictive modeling of gene expression reveals genetic, epistatic, and gene-by-environment interactions driving trait variation in switchgrass"**  

---

## 🔍 Abstract

*Panicum virgatum* (switchgrass) is a perennial bioenergy crop showing complex phenotypic variation across genotypes and environments. In this study, we integrated genomic and transcriptomic data to model trait variation and dissect genetic architecture using explainable machine learning.

We developed a predictive framework to:

- Train models using SNPs, gene expression (TPM), or combined features  
- Interpret predictions via SHAP values to identify key genetic, GxE, and epistatic interactions  
- Assess feature stability across environments (TX, MI) and model trait plasticity

**Key findings:**

- SHAP-based models identify transcripts predictive of traits and their plasticity across environments
- Gene-by-environment interactions contribute to trait variation  
- Epistatic SHAP interaction reveals higher-order structure beyond additive effects  

---

## 🗂 Repository Organization

### 📁 `1_SNP_calling/`  
Pipeline for calling SNPs from WGS data  
| Script | Description |
|--------|-------------|
| `0_genomeindex.sh` – `7_concatenate.sh` | From genome indexing to final GVCF/VCF generation using GATK |

---

### 📁 `2_eCor_kinship_h2/`  
Heritability estimation using kinship and environment-correlated expression data  
| File | Description |
|------|-------------|
| `eCor_Kinship_heritability.r` | R script to compute heritability, kinship matrix, and expression correlation profiles |

---

### 📁 `3_models/`  
Machine learning scripts for predictive modeling  
| Script | Description |
|--------|-------------|
| `1_xGBoost_SNP.py` | Model using SNPs only |
| `1_xGBoost_T.py` | Model using transcriptomic (TPM) data only |
| `1_xGBoost_SNP_T.py` | Model using combined SNP + TPM features |
| `2_xGBoost_*_featselection.py` | Scripts for feature selection and SHAP analysis |
| `3_xGBoost_diff_*_shapinteraction_matrix_10runs.py` | SHAP interaction matrix generation over 10 reps |
| `4_BRR_SNP_T_SNPT.r` | BRR models using SNP, TPM, and combined SNP+TPM features |

---

### 📁 `4_Genespace/`  
Gene-level annotation and orthology  
| File | Description |
|------|-------------|
| `01_orthofinder.sh` | OrthoFinder setup for gene orthology inference |
| `02_create_bed_files.r` | Create BED files for gene model mapping |
| `03_genespace_hpcc.r` | Annotate gene space using HPCC scripts |
| `04_genespace_hpcc.sh` | HPC-compatible wrapper |

---

## 📦 Requirements

### Python
- Python ≥ 3.8  
- Packages: `xgboost`, `shap`, `pandas`, `numpy`, `scikit-learn`


