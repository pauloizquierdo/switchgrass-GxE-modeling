######################## Script: 0 - Reference Preparation ########################
# Description:
# This script prepares the reference genome for SNP calling using BWA, SAMtools,
# and GATK. It unzips the FASTA, indexes it for alignment and variant calling,
# and generates required index and dictionary files.
####################################################################################################

######################## Load Required Modules ########################
module load BWA/0.7.17-20220923-GCCcore-12.3.0
module load SAMtools/1.18-GCC-12.3.0
module load GATK/4.5.0.0-GCCcore-12.3.0-Java-17

######################## Unzip and Prepare Reference FASTA ########################

# Unzip the reference genome
# NOTE: If already unzipped, this line can be skipped or replaced with a check
gunzip -c Pvirgatumvar_AP13HAP1_772_v6.0.hardmasked.fa.gz > Pvirgatumvar_AP13HAP1_772_v6.0.hardmasked.fa

######################## Indexing for Read Alignment and SNP Calling ########################

# BWA index: required for read alignment
bwa index Pvirgatumvar_AP13HAP1_772_v6.0.hardmasked.fa

# SAMtools index: required for GATK and general FASTA operations
samtools faidx Pvirgatumvar_AP13HAP1_772_v6.0.hardmasked.fa

# GATK sequence dictionary: required for variant calling and many GATK tools
gatk CreateSequenceDictionary \
    -R Pvirgatumvar_AP13HAP1_772_v6.0.hardmasked.fa \
    -O Pvirgatumvar_AP13HAP1_772_v6.0.hardmasked.dict

####################################################################################################
# End of Script
