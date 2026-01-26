#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20       
#SBATCH --mem-per-cpu=20G
#SBATCH --time=04:00:00
#SBATCH --job-name=sw_versions
#SBATCH -A data-machine
#SBATCH --output=genespace_swVersions_%A_%a.txt


module load OrthoFinder/2.5.5-foss-2023a # Load orthoFinder 2.5, genespace does not work with v3

for f in *fa ; do python  ../OrthoFinder/tools/primary_transcript.py $f ; done # Extract the longest transcript variant per gene 

Rscript genespace_hpcc.r > genespace_versions_Rlog.txt 2>&1

scontrol show job $SLURM_JOB_ID > genespace_Versions_${SLURM_JOB_ID}.txt