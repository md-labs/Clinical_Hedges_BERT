#!/bin/bash
#SBATCH -N 4
#SBATCH -n 28

#SBATCH -J BERT_Agave_Opioid
#SBATCH -o BERT_Agave.OUT
#SBATCH -e BERT_Agave.ERROR

#SBATCH -t 1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aambalav@asu.edu

export OMP_NUM_THREADS=28
#module load anaconda3/4.4.0
module load anaconda3/5.3.0
pip install --user -U pubmed-lookup

python3 src/Pubmed_Article_Extract_Scraping.py
