#########################
## KL Anneal: Cyclical ##
#########################
python main.py \
	--cuda \
	--train \
	--batch_size 20 \
	--tfr_decay_step 0.01 \
	--kl_anneal_cyclical \
	--exp_name cyclical-bs20

#python main.py --train \
#	--cuda \
#	--tfr_start_decay_epoch 0 \
#	--kl_anneal_cyclical \
#	--exp_name debug

#python main.py --test \
#	--cuda \
#	--tfr_start_decay_epoch 0 \
#	--kl_anneal_cyclical \
#	--exp_name cyclical-bs20 \
#	--model_dir ../logs/fp/cyclical-bs20

##########################
## KL Anneal: Monotonic ##
##########################
#python main.py \
#	--cuda \
#	--train \
#	--batch_size 20 \
#	--tfr_decay_step 0.01 \
#	--exp_name monotonic-bs20

##############
## Plotting ##
##############
#python others/utils.py