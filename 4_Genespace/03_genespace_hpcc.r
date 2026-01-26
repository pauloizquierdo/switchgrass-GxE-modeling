####################### 
# 1. Load Required Libraries
########################
library(GENESPACE)  # For genome synteny, orthology, and collinearity analysis
library(tidyverse)  # For general data manipulation (includes dplyr, readr, purrr)

# GENESPACE documentation: https://htmlpreview.github.io/?https://github.com/jtlovell/tutorials/blob/main/genespaceGuide.html
# MCScanX source: https://github.com/wyp1125/MCScanX/tree/master

########################
# 2. Set Working Directories
########################
# Define the GENESPACE output folder — must contain subfolders:
#   - "peptide/" with FASTA files of translated protein sequences (.faa)
#   - "bed/" with BED-format gene coordinates
wd <- "genespace_sw/"

# Path to the MCScanX binary (make sure it’s compiled and executable)
path2mcscanx <- "MCScanX-1.0.0/"  

########################
# 3. Initialize GENESPACE Configuration
########################

gpar <- init_genespace(
  wd = wd,                     # Working directory for GENESPACE
  path2mcscanx = path2mcscanx, # Path to MCScanX executable
  dotplots = "never",          # Skip dotplot generation (saves time)
  nCores = 20                  # Number of CPU cores to use
)

########################
# 4. Run GENESPACE
########################

# Execute the analysis pipeline: 
# - OrthoFinder or DIAMOND for orthologs
# - MCScanX for collinearity
# - Generates syntenic orthogroups and regions
out <- run_genespace(gpar, overwrite = TRUE)