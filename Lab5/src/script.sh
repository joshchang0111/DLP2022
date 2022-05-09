#########################
## KL Anneal: Cyclical ##
#########################
#python main.py \
#	--cuda \
#	--train \
#	--batch_size 20 \
#	--tfr_decay_step 0.01 \
#	--tfr_start_decay_epoch 150 \
#	--kl_anneal_cyclical \
#	--exp_name cyclical

#python main.py \
#	--cuda \
#	--test \
#	--test_set test \
#	--batch_size 20 \
#	--kl_anneal_cyclical \
#	--exp_name cyclical \
#	--model_dir ../logs/fp/cyclical

##########################
## KL Anneal: Monotonic ##
##########################
#python main.py \
#	--cuda \
#	--train \
#	--batch_size 20 \
#	--tfr_decay_step 0.01 \
#	--exp_name monotonic-bs20

#python main.py \
#	--cuda \
#	--test \
#	--test_set test \
#	--batch_size 20 \
#	--kl_anneal_cyclical \
#	--exp_name monotonic \
#	--model_dir ../logs/fp/monotonic

##############
## Plotting ##
##############
python others/utils.py