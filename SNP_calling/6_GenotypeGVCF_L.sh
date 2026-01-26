######################## Script: 06 - GenotypeGVCFs (Hardmasked Reference) ########################
# Description:
# This SLURM script runs GATK GenotypeGVCFs on intervals from a GenomicsDB workspace. 
# It is part of the joint genotyping process, producing population-level VCF files for each interval.
####################################################################################################

######################## SLURM Job Settings ########################
#SBATCH --job-name=GVCF_L
#SBATCH --nodes=1
#SBATCH --array=1:100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40GB
#SBATCH --time=30:00:00
#SBATCH -A glbrc
#SBATCH --output=GenotypeGVCFsL_%A_%a.txt
#SBATCH --constraint="[intel18|amd20|amd22]"

######################## Log Job Info ########################
echo "Starting job on $(hostname) at $(date)"

######################## Load Required Modules ########################
module purge
module load GCC/7.3.0-2.30 OpenMPI/3.1.1
module load GATK/4.1.4.1-Python-3.6.6

######################## Define Directories and Interval ########################
REFGEN_DIR="/referenceGenome"
POP_DIR="/population_masked"
TMP_DIR="/tpm_gatk"

# Extract interval from file
INT=$(awk "NR==${SLURM_ARRAY_TASK_ID}" intervals_5M.txt | cut -f4)

######################## Run GATK GenotypeGVCFs ########################
gatk --java-options "-Xms35G -Xmx35G -XX:ParallelGCThreads=2" GenotypeGVCFs \
    -R $REFGEN_DIR/Pvirgatumvar_AP13HAP1_772_v6.0.hardmasked.fa \
    -V gendb://GenomicsDB_L/$INT \
    -L $INT \
    --tmp-dir $TMP_DIR \
    -O $POP_DIR/${INT}_pop_hardmasked.vcf.gz

######################## Save Job Info ########################
scontrol show job $SLURM_JOB_ID > ${INT}p_${SLURM_ARRAY_TASK_ID}_hardmasked.txt

####################################################################################################
# End of Script
