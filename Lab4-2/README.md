# Lab4-2: Diabetic Retinopathy Detection
## Dataset
Download the complete dataset from the [link](https://drive.google.com/file/d/1GGD4t8mXhqMyvLhQ2a5VQ5o_ekKMDnLT/view?usp=sharing). Unzip this file and put the folder into `Lab4-2/dataset`.
## Dependencies
```
ipdb
tqdm
numpy
pandas
sklearn
matplotlib
Pillow
torch==1.6.0+cu101
torchvision==0.7.0+cu101
```
Install pytorch as follows.
```
$ pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
## Usage
Here is an example on how to train the models.
```
for experiment in "model=resnet18 bs=32" "model=resnet50 bs=16"
do
	eval $experiment

	python main.py \
		--train \
		--seed 123 \
		--model "$model" \
		--bs "$bs" \
		--save_exp

	CUDA_VISIBLE_DEVICES=1 python main.py \
		--train \
		--seed 123 \
		--model "$model" \
		--pretrained \
		--bs "$bs" \
		--save_exp
done
```