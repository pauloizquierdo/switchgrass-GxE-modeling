######################## Script: 07 - Concatenate VCF Files by Chromosome ########################
# Description:
# This SLURM script concatenates multiple per-interval VCF files into a single
# chromosome-level VCF file using GATK GatherVcfs. It then indexes the final VCF
# and converts it to BCF format using bcftools.
####################################################################################################

######################## SLURM Job Settings ########################
#SBATCH --job-name=concate
#SBATCH --array=1-18
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH -A data-machine
#SBATCH --output=concatenate_%A_%a.txt
#SBATCH --constraint="[intel18|amd20|amd22]"

######################## Log Job Info ########################
echo "Starting job on $(hostname) at $(date)"

######################## Load Required Modules ########################
module purge
module load GCC/7.3.0-2.30 OpenMPI/3.1.1
module load GATK/4.1.4.1-Python-3.6.6

######################## Determine Chromosome and Intervals ########################
# Get chromosome name from task ID
CHR=$(awk "NR==${SLURM_ARRAY_TASK_ID}" ../chromosomes.txt | cut -d " " -f1)

# Get all intervals for the chromosome
index=$(grep ${CHR} ../intervals_5M.txt | cut -f4)

######################## Build GATK GatherVcfs Command ########################
gatk_cmd="gatk GatherVcfs "

# Append each interval-specific VCF to the command
while IFS= read -r line; do
    interval=$(echo $line | tr -d '\n')
    gatk_cmd+="-I ${interval}_pop.vcf.gz "
done <<< "$index"

# Define output VCF for the chromosome
gatk_cmd+="-O ${CHR}_pop.vcf.gz"

# Execute the command
echo "Executing command: $gatk_cmd"
eval $gatk_cmd

# Index the final VCF
gatk IndexFeatureFile -I ${CHR}_pop.vcf.gz

######################## Convert VCF to BCF ########################
module purge
module load GCC/6.4.0-2.28 OpenMPI/2.1.2
module load bcftools/1.9.64

bcftools view -O b -o ${CHR}_pop.bcf ${CHR}_pop.vcf.gz

######################## Save Job Info ########################
scontrol show job $SLURM_JOB_ID > ${CHR}_job_info_concatenate.txt

####################################################################################################
# End of Script