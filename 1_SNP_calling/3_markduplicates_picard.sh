######################## Script: 03 - Mark Duplicates with Picard ########################
# Description:
# This SLURM batch script uses Picard to identify and mark PCR duplicates
# in sorted BAM files. It supports parallel execution via SLURM job arrays.
# Output includes deduplicated BAMs and duplication metrics for each sample.
####################################################################################################

######################## SLURM Job Settings ########################
#SBATCH --job-name=markdup_picard
#SBATCH --array=1-100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --time=24:00:00
#SBATCH --output=out_markdup_%A_%a.txt
#SBATCH -A data-machine
#SBATCH --constraint="[intel18|amd20|amd22]"

######################## Define Directories ########################
BAM_DIR="/1_mapping_sortedbams"
MARK_DIR="/markduplicates_picard"
TMP_DIR="/tpm_gatk"

######################## Log Job Info ########################
echo "Starting job on $(hostname) at $(date)"

######################## Load Required Modules ########################
module purge
module load picard/2.25.1-Java-11

######################## Parse Sample Info ########################
# Get sample name from samples.txt using SLURM array ID
SAMPLE_NAME=$(awk "NR==${SLURM_ARRAY_TASK_ID}" ../samples.txt | cut -d " " -f1)

######################## Run Picard MarkDuplicates ########################
java -Xms60G -Xmx60G -jar $EBROOTPICARD/picard.jar MarkDuplicates \
    I=$BAM_DIR/${SAMPLE_NAME}_sorted.bam \
    O=$MARK_DIR/${SAMPLE_NAME}_sorted_dedup.bam \
    M=$MARK_DIR/${SAMPLE_NAME}_sorted_dedup_metrics.txt \
    TMP_DIR=$TMP_DIR

######################## Save Job Info ########################
scontrol show job $SLURM_JOB_ID > ${SAMPLE_NAME}_info_markdup_${SLURM_ARRAY_TASK_ID}.txt

####################################################################################################
# End of Script