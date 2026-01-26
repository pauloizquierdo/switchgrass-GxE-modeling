######################## Script: 02 - BWA Read Alignment ########################
# Description:
# This SLURM batch script performs read alignment using BWA-MEM2 for paired-end
# or single-end trimmed reads. It handles multiple samples via SLURM array jobs,
# adds read group info, and outputs BAM files.
####################################################################################################

######################## SLURM Job Settings ########################
#SBATCH --job-name=bwa_glbrc
#SBATCH --array=1-100     # Sample line numbers from samples.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=44
#SBATCH --mem=176GB
#SBATCH --time=08:00:00
#SBATCH --output=out_bwa_align_%A_%a.txt
#SBATCH -A glbrc
#SBATCH --constraint="[intel18|amd20|amd22]"

######################## Define Directories ########################
FASTQ_DIR="/0_filtered_reads"
MAP_DIR="/1_mapping_reads_ap13"
REFGEN_DIR="/refGenome_switchgrass"

######################## Parse Sample Info ########################
# Extract sample name and type from line in samples.txt
SAMPLE_NAME=$(awk "NR==${SLURM_ARRAY_TASK_ID}" samples.txt | cut -d " " -f1)
SAMPLE_TYPE=$(awk "NR==${SLURM_ARRAY_TASK_ID}" samples.txt | cut -d " " -f2)

# Log host info
echo "Starting job on $(hostname) at $(date)"

######################## Load Required Modules ########################
module purge
module load bwa-mem2/2.2.1
module load GCC/9.3.0
module load SAMtools/1.15

######################## Run BWA-MEM2 Alignment ########################
if [[ $SAMPLE_TYPE == "PE" ]]; then
    # Paired-end alignment
    bwa-mem2 mem -t 32 -M \
        -R "@RG\tID:${SAMPLE_NAME}\tSM:${SAMPLE_NAME}\tLB:${SAMPLE_NAME}\tPL:ILLUMINA" \
        $REFGEN_DIR/Pvirgatumvar_AP13HAP1_772_v6.0.fa \
        $FASTQ_DIR/${SAMPLE_NAME}_trimmed_1.fastq.gz \
        $FASTQ_DIR/${SAMPLE_NAME}_trimmed_2.fastq.gz \
    | samtools view -@ 12 -S -b -o $MAP_DIR/${SAMPLE_NAME}.bam
else
    # Single-end alignment
    bwa-mem2 mem -t 32 -M \
        -R "@RG\tID:${SAMPLE_NAME}\tSM:${SAMPLE_NAME}\tLB:${SAMPLE_NAME}\tPL:ILLUMINA" \
        $REFGEN_DIR/Pvirgatumvar_AP13HAP1_772_v6.0.fa \
        $FASTQ_DIR/${SAMPLE_NAME}_trimmed.fastq.gz \
    | samtools view -@ 12 -S -b -o $MAP_DIR/${SAMPLE_NAME}.bam
fi

######################## Save Job Info ########################
scontrol show job $SLURM_JOB_ID > ${SAMPLE_NAME}_job_info_bwa_align_${SLURM_ARRAY_TASK_ID}.txt

####################################################################################################
# End of Script