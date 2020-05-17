#!/bin/bash
#SBATCH -N 1
#SBATCH -n 3
###SBATCH -p rcgpu4
###SBATCH -p rcgpu1
#SBATCH -p physicsgpu2
###SBATCH -p rcgpu1
###SBATCH -p mrlinegpu1
###SBATCH -p gpu
###SBATCH -p sulcgpu1
#SBATCH -q wildfire

#SBATCH --gres=gpu:4

#SBATCH -J BERT_Agave_CH_Ensemble
#SBATCH -o BERT_Agave_Ensemble.OUT
#SBATCH -e BERT_Agave_Ensemble.ERROR

#SBATCH -t 1-0:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aambalav@asu.edu

export OMP_NUM_THREADS=3
module load anaconda3/5.3.0

export PYTHONPATH=$PYTHONPATH:/home/aambalav/Clinical_Hedges
export PYTHONPATH=$PYTHONPATH:/home/aambalav/Clinical_Hedges/Cross_Validation_Experiments
export PYTHONPATH=$PYTHONPATH:/home/aambalav/Clinical_Hedges/Cross_Validation_Experiments/src/pytorch_pretrained_bert_Ensemble
export PRETRAINEDPATH=/home/aambalav/Clinical_Hedges/Cross_Validation_Experiments/BERT_Output_CV_ITL_Full_Min_Ensemble_PT

python run_classifier_CrossValidation_Ensemble.py --task_num=5 --bert_model=./scibert_scivocab_uncased --model_dir=$PRETRAINEDPATH/ --num_parts_start=0 --num_parts_end=1 --do_lower_case --task_name=clinicalhedges --data_dir=./CV_Data_Ensemble_Full_Min_PT --learning_rate=2e-5 --num_train_epochs=10 --output_dir=./BERT_Output_CV_Ensemble_Full_Min_PT/ --cache_dir=./BERT_CACHE --eval_batch_size=16 --max_seq_length=384 --train_batch_size=16 --do_eval

python run_classifier_CrossValidation_Ensemble.py --task_num=5 --bert_model=./scibert_scivocab_uncased --model_dir=$PRETRAINEDPATH/ --num_parts_start=1 --num_parts_end=2 --do_lower_case --task_name=clinicalhedges --data_dir=./CV_Data_Ensemble_Full_Min_PT --learning_rate=2e-5 --num_train_epochs=10 --output_dir=./BERT_Output_CV_Ensemble_Full_Min_PT/ --cache_dir=./BERT_CACHE --eval_batch_size=16 --max_seq_length=384 --train_batch_size=16 --do_eval

python run_classifier_CrossValidation_Ensemble.py --task_num=5 --bert_model=./scibert_scivocab_uncased --model_dir=$PRETRAINEDPATH/ --num_parts_start=2 --num_parts_end=3 --do_lower_case --task_name=clinicalhedges --data_dir=./CV_Data_Ensemble_Full_Min_PT --learning_rate=2e-5 --num_train_epochs=10 --output_dir=./BERT_Output_CV_Ensemble_Full_Min_PT/ --cache_dir=./BERT_CACHE --eval_batch_size=16 --max_seq_length=384 --train_batch_size=16 --do_eval

python run_classifier_CrossValidation_Ensemble.py --task_num=5 --bert_model=./scibert_scivocab_uncased --model_dir=$PRETRAINEDPATH/ --num_parts_start=3 --num_parts_end=4 --do_lower_case --task_name=clinicalhedges --data_dir=./CV_Data_Ensemble_Full_Min_PT --learning_rate=2e-5 --num_train_epochs=10 --output_dir=./BERT_Output_CV_Ensemble_Full_Min_PT/ --cache_dir=./BERT_CACHE --eval_batch_size=16 --max_seq_length=384 --train_batch_size=16 --do_eval

python run_classifier_CrossValidation_Ensemble.py --task_num=5 --bert_model=./scibert_scivocab_uncased --model_dir=$PRETRAINEDPATH/ --num_parts_start=4 --num_parts_end=5 --do_lower_case --task_name=clinicalhedges --data_dir=./CV_Data_Ensemble_Full_Min_PT --learning_rate=2e-5 --num_train_epochs=10 --output_dir=./BERT_Output_CV_Ensemble_Full_Min_PT/ --cache_dir=./BERT_CACHE --eval_batch_size=16 --max_seq_length=384 --train_batch_size=16 --do_eval

python run_classifier_CrossValidation_Ensemble.py --task_num=5 --bert_model=./scibert_scivocab_uncased --model_dir=$PRETRAINEDPATH/ --num_parts_start=5 --num_parts_end=6 --do_lower_case --task_name=clinicalhedges --data_dir=./CV_Data_Ensemble_Full_Min_PT --learning_rate=2e-5 --num_train_epochs=10 --output_dir=./BERT_Output_CV_Ensemble_Full_Min_PT/ --cache_dir=./BERT_CACHE --eval_batch_size=16 --max_seq_length=384 --train_batch_size=16 --do_eval

python run_classifier_CrossValidation_Ensemble.py --task_num=5 --bert_model=./scibert_scivocab_uncased --model_dir=$PRETRAINEDPATH/ --num_parts_start=6 --num_parts_end=7 --do_lower_case --task_name=clinicalhedges --data_dir=./CV_Data_Ensemble_Full_Min_PT --learning_rate=2e-5 --num_train_epochs=10 --output_dir=./BERT_Output_CV_Ensemble_Full_Min_PT/ --cache_dir=./BERT_CACHE --eval_batch_size=16 --max_seq_length=384 --train_batch_size=16 --do_eval

python run_classifier_CrossValidation_Ensemble.py --task_num=5 --bert_model=./scibert_scivocab_uncased --model_dir=$PRETRAINEDPATH/ --num_parts_start=7 --num_parts_end=8 --do_lower_case --task_name=clinicalhedges --data_dir=./CV_Data_Ensemble_Full_Min_PT --learning_rate=2e-5 --num_train_epochs=10 --output_dir=./BERT_Output_CV_Ensemble_Full_Min_PT/ --cache_dir=./BERT_CACHE --eval_batch_size=16 --max_seq_length=384 --train_batch_size=16 --do_eval

python run_classifier_CrossValidation_Ensemble.py --task_num=5 --bert_model=./scibert_scivocab_uncased --model_dir=$PRETRAINEDPATH/ --num_parts_start=8 --num_parts_end=9 --do_lower_case --task_name=clinicalhedges --data_dir=./CV_Data_Ensemble_Full_Min_PT --learning_rate=2e-5 --num_train_epochs=10 --output_dir=./BERT_Output_CV_Ensemble_Full_Min_PT/ --cache_dir=./BERT_CACHE --eval_batch_size=16 --max_seq_length=384 --train_batch_size=16 --do_eval

python run_classifier_CrossValidation_Ensemble.py --task_num=5 --bert_model=./scibert_scivocab_uncased --model_dir=$PRETRAINEDPATH/ --num_parts_start=9 --num_parts_end=10 --do_lower_case --task_name=clinicalhedges --data_dir=./CV_Data_Ensemble_Full_Min_PT --learning_rate=2e-5 --num_train_epochs=10 --output_dir=./BERT_Output_CV_Ensemble_Full_Min_PT/ --cache_dir=./BERT_CACHE --eval_batch_size=16 --max_seq_length=384 --train_batch_size=16 --do_eval
