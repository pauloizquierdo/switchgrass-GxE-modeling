######################## 
# 1. Load libraries
library(tidyverse)  
########################

# References:
# GENESPACE documentation: https://htmlpreview.github.io/?https://github.com/jtlovell/tutorials/blob/main/genespaceGuide.html
# MCScanX (synteny engine): https://github.com/wyp1125/MCScanX/tree/master

########################
# 1.  List all GFF3 files in the current directory
########################

# Set the working directory where your GFF3 files are located
gff_files <- list.files(pattern = "\\.gff3$", full.names = TRUE)

# Define the directory to save BED files
bed_dir <- "primary_transcripts"

########################
# 2. Define a function to convert a GFF3 file to BED format
########################

gff_to_bed <- function(file_path) {
  # Read the GFF3 file, skipping comments
  gff <- read_tsv(file_path, comment = "#", col_names = FALSE)
  
  # Assign column names based on GFF3 specification
  colnames(gff) <- c("chr", "source", "type", "start", "end", "score", "strand", "phase", "attributes")
  
  # Extract gene features and clean up gene IDs
  genes <- gff %>%
    filter(type == "gene") %>%
    mutate(
      gene_id = str_extract(attributes, "ID=[^;]+"),         # Extract the 'ID' field
      gene_id = str_replace(gene_id, "ID=", ""),             # Remove 'ID=' prefix
      gene_id = str_remove(gene_id, "\\.v[0-9]+\\.[0-9]+$")  # Remove version suffix (e.g., .v6.1)
    ) %>%
    mutate(start = start - 1) %>%  # Convert start coordinate to 0-based (BED format)
    select(chr, start, end, gene_id)  # Select BED-compatible columns
  
  # Save the output as a BED file with the same base name
  bed_filename <- paste0(basename(file_path) %>% str_remove("\\.gff3$"), ".bed")
  write_tsv(genes, file.path(bed_dir, bed_filename), col_names = FALSE)
}

########################
# 3. Apply the conversion function to all GFF3 files
########################
walk(gff_files, gff_to_bed)