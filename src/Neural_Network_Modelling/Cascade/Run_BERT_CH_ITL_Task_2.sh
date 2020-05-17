#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
###SBATCH -p rcgpu2
###SBATCH -p rcgpu3
#SBATCH -p rcgpu6
###SBATCH -p wzhengpu1
###SBATCH -p physicsgpu1
#SBATCH -q wildfire

#SBATCH --gres=gpu:2

#SBATCH -J BERT_Agave_CH_ITL_Task_2_M
#SBATCH -o BERT_Agave_ITL_Task_2.OUT
#SBATCH -e BERT_Agave_ITL_Task_2.ERROR

#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aambalav@asu.edu

export OMP_NUM_THREADS=2
###module load anaconda3/4.4.0
module load anaconda3/5.3.0

pip install --user pytorch-pretrained-bert==0.6.0

python run_classifier_CrossValidation_Task.py --num_parts_start=0 --num_parts_end=10  --task_num=2 --model_file=./scibert_scivocab_uncased --bert_model=./scibert_scivocab_uncased --do_lower_case --task_name=clinicalhedges --data_dir=./CV_Data_ITL_Marshall_Cascade --learning_rate=2e-5 --num_train_epochs=10 --output_dir=./BERT_Output_CV_ITL_Cascade_Marshall/ --cache_dir=./BERT_CACHE --eval_batch_size=16 --max_seq_length=384 --train_batch_size=16 --do_train --do_eval
