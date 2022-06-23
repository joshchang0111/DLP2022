########################
## Data Preprocessing ##
########################
#python data/preprocess.py --preprocess
#python data/preprocess.py --split_5_fold
#python data/preprocess.py --split_target
#python data/statistics.py

##########
## Main ##
##########
## Original examples from huggingface
#python main.py \
#	--model_name_or_path bert-base-uncased \
#	--task_name mrpc \
#	--do_train \
#	--do_eval \
#	--max_seq_length 128 \
#	--per_device_train_batch_size 32 \
#	--learning_rate 2e-5 \
#	--num_train_epochs 3 \
#	--output_dir ../result/mrpc \
#	--overwrite_output_dir

## Train on my stance dataset
#models="bert-base-uncased"
#models="roberta-base allenai/longformer-base-4096 google/bigbird-roberta-base"
## 5-Fold
for i in $(seq 0 4)
do
	python main.py \
		--model_name_or_path roberta-base \
		--dataset_name IBM \
		--fold "$i" \
		--train_file train.csv \
		--validation_file test.csv \
		--do_train \
		--do_eval \
		--max_seq_length 128 \
		--per_device_train_batch_size 32 \
		--learning_rate 2e-5 \
		--num_train_epochs 10 \
		--output_dir /mnt/hdd1/projects/DLPFP/result \
		--overwrite_output_dir \
		--save_model_accord_to_metric

	python main.py \
		--model_name_or_path google/bigbird-roberta-base \
		--dataset_name IBM \
		--fold "$i" \
		--train_file train.csv \
		--validation_file test.csv \
		--do_train \
		--do_eval \
		--max_seq_length 128 \
		--per_device_train_batch_size 32 \
		--learning_rate 2e-5 \
		--num_train_epochs 10 \
		--output_dir /mnt/hdd1/projects/DLPFP/result \
		--overwrite_output_dir \
		--save_model_accord_to_metric

	python main.py \
		--model_name_or_path allenai/longformer-base-4096 \
		--dataset_name IBM \
		--fold "$i" \
		--train_file train.csv \
		--validation_file test.csv \
		--do_train \
		--do_eval \
		--max_seq_length 128 \
		--per_device_train_batch_size 4 \
		--learning_rate 2e-5 \
		--num_train_epochs 10 \
		--output_dir /mnt/hdd1/projects/DLPFP/result \
		--overwrite_output_dir \
		--save_model_accord_to_metric
done

## Target-wise
for i in $(seq 0 4)
do
	python main.py \
		--model_name_or_path roberta-base \
		--dataset_name IBM \
		--fold "$i" \
		--target_wise \
		--train_file train.csv \
		--validation_file test.csv \
		--do_train \
		--do_eval \
		--max_seq_length 128 \
		--per_device_train_batch_size 32 \
		--learning_rate 2e-5 \
		--num_train_epochs 10 \
		--output_dir /mnt/hdd1/projects/DLPFP/result \
		--overwrite_output_dir \
		--save_model_accord_to_metric

	python main.py \
		--model_name_or_path google/bigbird-roberta-base \
		--dataset_name IBM \
		--fold "$i" \
		--target_wise \
		--train_file train.csv \
		--validation_file test.csv \
		--do_train \
		--do_eval \
		--max_seq_length 128 \
		--per_device_train_batch_size 32 \
		--learning_rate 2e-5 \
		--num_train_epochs 10 \
		--output_dir /mnt/hdd1/projects/DLPFP/result \
		--overwrite_output_dir \
		--save_model_accord_to_metric

	python main.py \
		--model_name_or_path allenai/longformer-base-4096 \
		--dataset_name IBM \
		--fold "$i" \
		--target_wise \
		--train_file train.csv \
		--validation_file test.csv \
		--do_train \
		--do_eval \
		--max_seq_length 128 \
		--per_device_train_batch_size 4 \
		--learning_rate 2e-5 \
		--num_train_epochs 10 \
		--output_dir /mnt/hdd1/projects/DLPFP/result \
		--overwrite_output_dir \
		--save_model_accord_to_metric
done

## Evaluation only
#python main.py \
#	--model_name_or_path bert-base-uncased \
#	--dataset_name IBM \
#	--train_file train.csv \
#	--validation_file test.csv \
#	--do_eval \
#	--max_seq_length 128 \
#	--per_device_train_batch_size 32 \
#	--best_checkpoint_path ../result/IBM/checkpoint-540
