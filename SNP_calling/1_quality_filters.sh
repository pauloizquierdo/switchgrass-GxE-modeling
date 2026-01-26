######################## Script: 01 - Read Filtering with fastp ########################
# Description:
# This script performs quality filtering and adapter trimming of paired-end FASTQ files
# using fastp. It processes all raw reads in a given directory and outputs filtered reads,
# along with HTML and JSON reports for each sample.
####################################################################################################

######################## SLURM Job Settings ########################
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=fastp_processing
#SBATCH --time=03:30:00
#SBATCH --output=fastp_processing_%j.out
#SBATCH --error=fastp_processing_%j.err

######################## Environment Setup ########################
# Activate the conda environment that contains fastp
conda activate snp_calling

######################## Directory Setup ########################
# Define input and output directories
input_dir="../raw_data/"
output_dir="/0_filtered_reads"
mkdir -p "$output_dir"

######################## Processing FASTQ Files ########################
# Loop through all R1 FASTQ files and find corresponding R2 pairs
for r1 in ${input_dir}/*_R1_*.fastq.gz; do
    r2="${r1/_R1/_R2}"  # Infer R2 filename by string substitution
    sample_name=$(basename "$r1" | sed 's/_R1_.*//')  # Extract sample name

    echo "Processing sample: $sample_name"

    # Run fastp for filtering and trimming
    fastp -i "$r1" \
          -I "$r2" \
          -o "${output_dir}/${sample_name}_trimmed_1.fastq.gz" \
          -O "${output_dir}/${sample_name}_trimmed_2.fastq.gz" \
          --thread 6 \
          -j "${output_dir}/${sample_name}_fastp.json" \
          -h "${output_dir}/${sample_name}_fastp.html" \
          > "${output_dir}/${sample_name}_fastp.log" 2>&1

done

######################## Completion Message ########################
echo "Processing sample: $sample_name"
echo "All samples processed. Filtered files and reports are in $output_dir."

####################################################################################################
# End of Script