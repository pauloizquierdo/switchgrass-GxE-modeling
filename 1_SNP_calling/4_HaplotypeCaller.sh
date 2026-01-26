######################## Script: 04 - HaplotypeCaller GVCF Mode ########################
# Description:
# This SLURM batch script runs GATK HaplotypeCaller in GVCF mode for a set of
# BAM files (post-mark-duplicates). It processes samples in parallel using SLURM
# job arrays and targets a specific chromosome region.
####################################################################################################

######################## SLURM Job Settings ########################
#SBATCH --job-name=haplotypecaller
#SBATCH --array=1-100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60GB
#SBATCH --time=4-00:00:00
#SBATCH --output=out_haplocall_1_2_%A_%a.txt
#SBATCH -A data-machine
#SBATCH --constraint="[intel18|amd20|amd22]"

######################## Define Directories ########################
MARK_DIR="/markduplicates_picard"
GENO_DIR="/genotyping_pi"
TMP_DIR="/tpm_gatk"
REFGEN_DIR="/refGenome_switchgrass"

######################## Log Job Info ########################
echo "Starting job on $(hostname) at $(date)"

######################## Load Required Modules ########################
module purge
module load GCC/7.3.0-2.30 OpenMPI/3.1.1
module load GATK/4.1.4.1-Python-3.6.6

######################## Parse Sample Info ########################
SAMPLE_NAME=$(awk "NR==${SLURM_ARRAY_TASK_ID}" ../samples.txt | cut -d " " -f1)

######################## Run GATK HaplotypeCaller ########################
gatk --java-options "-Xms40g -Xmx40g -XX:ParallelGCThreads=2" HaplotypeCaller \
    -R $REFGEN_DIR/Pvirgatumvar_AP13HAP1_772_v6.0.fa \
    -I $MARK_DIR/${SAMPLE_NAME}_sorted_dedup.bam \
    -O $GENO_DIR/${SAMPLE_NAME}_sorted_dedup_chr1-2N.vcf.gz \
    -ERC GVCF \
    --tmp-dir $TMP_DIR \
    -L Chr01N

######################## Save Job Info ########################
scontrol show job $SLURM_JOB_ID > ${SAMPLE_NAME}_job_info_genotyping_chr1N.txt

####################################################################################################
# End of Script