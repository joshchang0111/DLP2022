###########
## Debug ##
###########
#python main.py \
#	--train \
#	--seed 123 \
#	--model resnet18 \
#	--bs 32 \

#python main.py \
#	--test \
#	--seed 123 \
#	--model resnet18 \
#	--bs 32 \
#	--save_confusion_matrix \

#######################
## Basic Experiments ##
#######################
## Train ##
#for experiment in "model=resnet18 bs=32" "model=resnet50 bs=16"
#do
#	eval $experiment
#
#	#python main.py \
#	#	--train \
#	#	--seed 123 \
#	#	--model "$model" \
#	#	--bs "$bs" \
#	#	--save_exp
#
#	CUDA_VISIBLE_DEVICES=1 python main.py \
#		--train \
#		--seed 123 \
#		--model "$model" \
#		--pretrained \
#		--bs "$bs" \
#		--save_exp
#done

## Test ##
#for model in resnet18 resnet50
#do
#	python main.py \
#		--test \
#		--seed 123 \
#		--model "$model" \
#		--bs 32 \
#		--save_confusion_matrix
#
#	python main.py \
#		--test \
#		--seed 123 \
#		--model "$model" \
#		--pretrained \
#		--bs 32 \
#		--save_confusion_matrix
#done

#for model in resnet18 resnet50
#do
#	python main.py \
#		--test \
#		--seed 123 \
#		--model "$model" \
#		--bs 32 \
#		--save_confusion_matrix \
#		--result_path ../result/basic-normalize \
#		--checkpoint_path ../checkpoints/basic-normalize
#
#	python main.py \
#		--test \
#		--seed 123 \
#		--model "$model" \
#		--pretrained \
#		--bs 32 \
#		--save_confusion_matrix \
#		--result_path ../result/basic-normalize \
#		--checkpoint_path ../checkpoints/basic-normalize
#done

python main.py \
	--test \
	--seed 123 \
	--model resnet50 \
	--pretrained \
	--bs 32 \
	--checkpoint_path ../checkpoints/basic-normalize

###################
## Weighted Loss ##
###################
## from scratch
#for experiment in "model=resnet18 bs=32" "model=resnet50 bs=16"
#do
#	eval $experiment
#
#	CUDA_VISIBLE_DEVICES=0 python main.py \
#		--train \
#		--seed 123 \
#		--model "$model" \
#		--bs "$bs" \
#		--save_exp \
#		--loss_weight
#	
#	#python main.py \
#	#	--train \
#	#	--seed 123 \
#	#	--model "$model" \
#	#	--pretrained \
#	#	--bs "$bs" \
#	#	--save_exp \
#	#	--loss_weight
#done

##########################
## Plot Learning Curves ##
##########################
#python others/utils.py




