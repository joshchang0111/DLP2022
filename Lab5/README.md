# Lab5: Conditional VAE for Video Prediction
## Dependencies
```
ipdb
tqdm
gdown==4.3.0
numpy
scipy
imageio
matplotlib
scikit-image
mpl_axes_aligner
torch==1.6.0+cu101
torchvision==0.7.0+cu101
```
Install pytorch as follows.
```
$ pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
## Dataset
Run the following command to download the complete dataset. Unzip the file and put the folders into `Lab5/dataset`.
```
$ gdown https://drive.google.com/open?id=1t8iOfBUYDFQH65RNWIgsEYenAo-HktAu&authuser=0
```
## Usage
Train the model with following command.
```
python main.py \
	--cuda \
	--train \
	--batch_size 20 \
	--tfr_decay_step 0.01 \
	--tfr_start_decay_epoch 150 \
	--kl_anneal_cyclical \
	--exp_name cyclical
```
You can disable cyclical KL annealing by removing the argument `--kl_anneal_cyclical`. `--exp_name` is the directory name of your training results.