#########################################
## Construct new examples by ourselves ##
#########################################
#python demo.py \
#	--model_name_or_path /mnt/hdd1/projects/DLPFP/result \
#	--dataset_name IBM \
#	--fold 0 \
#	--validation_file test.csv \
#	--demo_mode construct_new \
#	--demo_model google/bigbird-roberta-base

## Target: Face_masks, Fauci, School_closures, Stay_at_home_orders
python demo.py \
	--model_name_or_path /mnt/hdd1/projects/DLPFP/result \
	--dataset_name covid19 \
	--fold 0 \
	--validation_file test.csv \
	--demo_mode construct_new \
	--demo_model google/bigbird-roberta-base

## Target: Hillary Clinton, Feminist Movement, Legalization of Abortion, Atheism, Climate Change is a Real Concern
#python demo.py \
#	--model_name_or_path /mnt/hdd1/projects/DLPFP/result \
#	--dataset_name semeval \
#	--fold 0 \
#	--validation_file test.csv \
#	--demo_mode construct_new \
#	--demo_model google/bigbird-roberta-base

############################################################
## Sample correct / wrong predictions from validation set ##
############################################################
#python demo.py \
#	--model_name_or_path /mnt/hdd1/projects/DLPFP/result \
#	--dataset_name IBM \
#	--fold 0 \
#	--train_file train.csv \
#	--validation_file test.csv \
#	--max_seq_length 128 \
#	--do_eval \
#	--demo_mode sample \
#	--demo_model google/bigbird-roberta-base

#python demo.py \
#	--model_name_or_path /mnt/hdd1/projects/DLPFP/result \
#	--dataset_name covid19 \
#	--fold 0 \
#	--train_file train.csv \
#	--validation_file test.csv \
#	--max_seq_length 128 \
#	--do_eval \
#	--demo_mode sample \
#	--demo_model google/bigbird-roberta-base

#python demo.py \
#	--model_name_or_path /mnt/hdd1/projects/DLPFP/result \
#	--dataset_name semeval \
#	--fold 0 \
#	--train_file train.csv \
#	--validation_file test.csv \
#	--max_seq_length 128 \
#	--do_eval \
#	--demo_mode sample \
#	--demo_model google/bigbird-roberta-base
