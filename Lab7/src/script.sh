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

#python main.py \
#	--train \
#	--n_eval 10 \
#	--lr_G 2e-4 \
#	--lr_D 2e-4 \
#	--gan_type cgan \
#	--exp_name cgan-d1g4

#python main.py \
#	--train \
#	--n_eval 10 \
#	--batch_size 128 \
#	--lr_G 1e-3 \
#	--lr_D 2e-4 \
#	--gan_type cgan \
#	--exp_name cgan-d1g4-2

## WGAN-GP+CGAN, acc = 0.6111
#python main.py \
#	--train \
#	--lr_G 2e-4 \
#	--lr_D 2e-4 \
#	--beta1 0 \
#	--beta2 0.9 \
#	--gan_type wgan \
#	--exp_name wgan-gp

## acc = 0.6389
#python main.py \
#	--train \
#	--n_eval 10 \
#	--lr_G 1e-4 \
#	--lr_D 1e-4 \
#	--beta1 0.5 \
#	--beta2 0.9 \
#	--gan_type wgan \
#	--exp_name wgan-gp-10-eval

##########
## Test ##
##########
## wgan, acc = 0.6111
#python main.py \
#	--test \
#	--gan_type wgan \
#	--checkpoint_epoch 166-0 \
#	--exp_name wgan-gp-0.6111

## wgan-gp-10-eval, test.json acc = 0.6398, new_test.json acc = 0.6429
#python main.py \
#	--test \
#	--gan_type wgan \
#	--checkpoint_epoch 278-100 \
#	--exp_name wgan-gp-10-eval \
#	--test_file new_test.json

## cgan-d1g4, test.json acc = 0.5833, new_test.json acc = 0.5714
python main.py \
	--test \
	--gan_type cgan \
	--checkpoint_epoch 271-50 \
	--exp_name cgan-d1g4 \
	--test_file new_test.json