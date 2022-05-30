###########
## Train ##
###########
## InfoGAN
#python main.py \
#	--train \
#	--lr_G 1e-3 \
#	--lr_D 2e-4 \
#	--lambda_Q 1 \
#	--c_dim 50 \
#	--exp_name InfoGAN

## CGAN
#python main.py \
#	--train \
#	--lr_G 1e-3 \
#	--lr_D 2e-4 \
#	--gan_type cgan \
#	--exp_name cgan

## WGAN-GP+CGAN
#python main.py \
#	--train \
#	--lr_G 2e-4 \
#	--lr_D 2e-4 \
#	--beta1 0 \
#	--beta2 0.9 \
#	--gan_type wgan \
#	--exp_name wgan-gp

python main.py \
	--train \
	--n_eval 10 \
	--lr_G 1e-4 \
	--lr_D 1e-4 \
	--beta1 0.5 \
	--beta2 0.9 \
	--gan_type wgan \
	--exp_name wgan-gp-10-eval

##########
## Test ##
##########
#python main.py \
#	--seed 14 \
#	--test \
#	--gan_type wgan \
#	--checkpoint_epoch 166-0 \
#	--exp_name wgan-gp-0.6111