#!/bin/sh


#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=16gb
#SBATCH -c 4
#SBATCH -a 87-94
#SBATCH -t 2-00:00:00  
#SBATCH -J bert_mhasan8
#SBATCH -o /scratch/mhasan8/output/bert_output%j
#SBATCH -e /scratch/mhasan8/output/bert_error%j
#SBATCH --mail-type=all    

module load anaconda3/5.3.0b
source activate wasifur
module load git


python /scratch/mhasan8/Multimodal_study/BERT_multimodal_transformer/bert_running_different_configs.py --dataset=mosi

