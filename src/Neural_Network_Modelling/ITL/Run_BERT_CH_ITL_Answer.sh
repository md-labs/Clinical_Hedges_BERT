#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
###SBATCH -p rcgpu3
###SBATCH -p physicsgpu1
#SBATCH -p cidsegpu2
###SBATCH -p sulcgpu2
#SBATCH -q wildfire

#SBATCH --gres=gpu:2

#SBATCH -J BERT_Agave_CH_ITL_Answer_1.5_0.4_D
#SBATCH -o BERT_Agave_ITL_Answer_PT.OUT
#SBATCH -e BERT_Agave_ITL_Answer_PT.ERROR

#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aambalav@asu.edu

export OMP_NUM_THREADS=2
###module load anaconda3/4.4.0
module load anaconda3/5.3.0

pip install --user pytorch-pretrained-bert==0.6.0

python run_classifier_CrossValidation_ITL.py --num_parts_start=0 --num_parts_end=10  --task_num=5 --model_file=./scibert_scivocab_uncased --bert_model=./scibert_scivocab_uncased --do_lower_case --task_name=clinicalhedges --data_dir=./CV_Data_ITL_Del_Fiol_PT_1.5_0.2 --learning_rate=2e-5 --num_train_epochs=10 --output_dir=./BERT_Output_CV_ITL_1.5_0.2_Del_Fiol/ --cache_dir=./BERT_CACHE --eval_batch_size=16 --max_seq_length=384 --train_batch_size=16 --do_train --do_eval
