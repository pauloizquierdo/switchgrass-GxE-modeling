# This script runs OrthoFinder on the P. virgatum (switchgrass) reference genomes.
# It identifies homeologous genes, extracting primary transcripts, and running orthology analysis.

########################
# Step 1: Prepare the Input Data
########################

# Download the protein (amino acid) sequences for the P. virgatum genomes from Phytozome.
# Website: https://phytozome-next.jgi.doe.gov/

# Decompress the downloaded protein file (.gz format) to obtain the raw FASTA file.

gzip -d Pvirgatum_273_v1.1.protein.fa.gz
gzip -d Pvirgatum_383_v3.1.protein.fa.gz
gzip -d Pvirgatum_450_v4.1.protein.fa.gz
gzip -d Pvirgatum_516_v5.1.protein.fa.gz
gzip -d Pvirgatumvar_AP13HAP1_772_v6.1.protein.fa.gz

########################
# Step 2: Prepare Files for OrthoFinder
########################

# Note: The protein files contain multiple transcript isoforms per gene.
# To avoid inflated orthogroups, we keep only the longest (primary) isoform per gene.
# This reduces computation time (~10x faster) and improves orthology accuracy.

# Use OrthoFinder's script (primary_transcript.py) to extract the primary transcript from each file.
# This creates new FASTA files containing only the longest isoform per gene.

for f in *fa ; do python  ../OrthoFinder/tools/primary_transcript.py $f ; done

########################
# Step 4: Run OrthoFinder
########################

# Load OrthoFinder module
module load OrthoFinder/2.5.5-foss-2023a

# Run OrthoFinder on the directory containing the primary transcript FASTA files.
# The '-f' option specifies the input folder.
orthofinder -f primary_transcripts/