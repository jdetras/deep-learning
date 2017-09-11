#!/bin/bash

#SBATCH --job-name=TensorFlowTest
#SBATCH --output TensorFlowTest.%j.out
#SBATCH --error TensorFlowTest.%j.error
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=16G
#SBATCH --time=07:00:00
#SBATCH --partition=batch
#SBATCH --mail-user=j.detras@irri.org
#SBATCH --mail-type=ALL

module load anaconda2

python snp2phenotype.py
