# Final Project - Stance Detection with Transformers
## Dependencies
```
ipdb
tqdm
pandas
scipy
sklearn

*** pytorch ***
torch==1.6.0+cu101

*** huggingface packages ***
datasets
transformers==4.19.0.dev0
```
Install pytorch as follows. (not necessary same version as mine)
```
$ pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
Install huggingface transformers **from source** as follows.
```
$ pip install git+https://github.com/huggingface/transformers
```

## Dataset
Download our processed datasets from [this link](https://drive.google.com/drive/folders/1GWY9H5jkaAPIyC93PNDDjiktkNMIu75o?usp=sharing). Put the datasets in the following directory `dataset/processed/$YOUR_DATASET_NAME` at the same level of `src`.

## Run the code
```
python main.py \
	--model_name_or_path bert-base-uncased \
	--dataset_name IBM \
	--fold $YOUR_FOLD_IDX \
	--train_file train.csv \
	--validation_file test.csv \
	--do_train \
	--do_eval \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 10 \
	--output_dir $YOUR_OUTPUT_DIR \
	--overwrite_output_dir \
	--save_model_accord_to_metric
```
To run experiments on target-wise dataset, remember to use the argument `--target_wise` and specify the correct fold index at `--fold`.

You can also write your command in the file `src/scripts/script.sh` and execute the following lines.
```
$ cd src
$ sh scripts/script.sh
```