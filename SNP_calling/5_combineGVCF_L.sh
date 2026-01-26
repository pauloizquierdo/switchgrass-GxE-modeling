######################## Script: 05 - GenomicsDBImport ########################
# Description:
# This SLURM script uses GATK GenomicsDBImport to combine multiple per-sample
# GVCF files into a GenomicsDB workspace for joint genotyping. It supports array
# jobs for processing genomic intervals in parallel.
####################################################################################################

######################## SLURM Job Settings ########################
#SBATCH --job-name=combine_GVCF_L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50GB
#SBATCH --time=12:00:00
#SBATCH -A glbrc
#SBATCH --output=combine_GVCF_L_%A_%a.txt
#SBATCH --constraint="[intel18|amd20|amd22]"

######################## Log Job Info ########################
echo "Starting job on $(hostname) at $(date)"

######################## Load Required Modules ########################
module purge
module load GCC/7.3.0-2.30 OpenMPI/3.1.1
module load GATK/4.1.4.1-Python-3.6.6

######################## Define Directories and Interval ########################
TMP_DIR="/tpm_gatk"
GENO_DIR="/genotyping_pi/"
GDBI="/genotyping_pi/GenomicsDB_L"

# Extract interval (e.g., 5 Mb region) from interval file
INT=$(awk "NR==${SLURM_ARRAY_TASK_ID}" intervals_5M.txt | cut -f4)

# Create a unique folder for the GenomicsDB workspace (if needed)
CHROMOSOME_DB_PATH="${GDBI}/${INT}"

######################## Run GenomicsDBImport ########################
gatk --java-options "-Xms40g -Xmx40g -XX:ParallelGCThreads=2" GenomicsDBImport \
  --sample-name-map samples_map.txt \
  --genomicsdb-workspace-path $CHROMOSOME_DB_PATH \
  --tmp-dir $TMP_DIR \
  -L $INT \
  --batch-size 100 \
  --reader-threads 8

######################## Save Job Info ########################
scontrol show job $SLURM_JOB_ID > ${INT}_GenomicsDBImport_L_job_info.txt

####################################################################################################
# End of Script