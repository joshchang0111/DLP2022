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
#	#	--lr 5e-4 \
#	#	--bs "$bs" \
#	#	--optim adam \
#	#	--save_exp
#
#	python main.py \
#		--train \
#		--seed 123 \
#		--model "$model" \
#		--pretrained \
#		--lr 5e-4 \
#		--bs "$bs" \
#		--optim adam \
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

python main.py \
	--test \
	--seed 123 \
	--model resnet50 \
	--pretrained \
	--bs 32 \
	--save_confusion_matrix \
	--checkpoint_path ../checkpoints/wo_loss_weight

###################
## Weighted Loss ##
###################
#for experiment in "model=resnet18 bs=32" "model=resnet50 bs=16"
#do
#	eval $experiment
#
#	python main.py \
#		--train \
#		--seed 123 \
#		--model "$model" \
#		--bs "$bs" \
#		--save_exp \
#		--loss_weight
#	
#	python main.py \
#		--train \
#		--seed 123 \
#		--model "$model" \
#		--pretrained \
#		--bs "$bs" \
#		--save_exp \
#		--loss_weight
#done

##########################
## Plot Learning Curves ##
##########################
#python others/utils.py




