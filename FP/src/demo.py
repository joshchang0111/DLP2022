import os
import sys
import ipdb
import logging
import numpy as np

import torch
import datasets
from datasets import load_dataset

import transformers
from transformers import (
	HfArgumentParser,
	TrainingArguments,
	set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

## Self-defined
from others.args import DataTrainingArguments, ModelArguments, CustomTrainingArguments, args_post_init
from others.processes import train_process, eval_process, predict_process
from others.utils import setup_logging
from data.build_datasets import build_datasets
from models.load_model import load_model
from pipelines.build_trainer import build_trainer

## Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.19.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)

def main():
	## Parse args
	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
	if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
		## If we pass only one argument to the script and it's the path to a json file,
		## let's parse it to get our arguments.
		model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
	else:
		model_args, data_args, training_args = parser.parse_args_into_dataclasses()
	model_args, data_args, training_args = args_post_init(model_args, data_args, training_args)

	## Setups
	last_checkpoint = setup_logging(logger, model_args, data_args, training_args)
	set_seed(training_args.seed) ## Set seed before initializing model.

	################
	## Load model ##
	################
	config, tokenizer, model = load_model(data_args, model_args, training_args)

	################
	## Demo Phase ##
	################
	if data_args.demo_mode == "construct_new":
		## Get labels
		data_files = {"validation": "{}/{}/split_{}/{}".format(data_args.dataset_root, data_args.dataset_name, data_args.fold, data_args.validation_file)}
		raw_datasets = load_dataset(
			"csv",
			data_files=data_files,
			cache_dir=model_args.cache_dir,
			use_auth_token=True if model_args.use_auth_token else None,
		)
		label_list = raw_datasets["validation"].unique("stance")
		label_list.sort()

		print("\nConstruct examples by ourselves...")
		while True:
			target = input("Type in your target (press enter to exit): ")
			if not target:
				break
			claim  = input("Type in your claim  (press enter to exit): ")
			if not claim:
				break

			inputs = tokenizer([[claim, target]], add_special_tokens=True, return_tensors="pt")

			with torch.no_grad():
				logits = model(**inputs).logits
			predicted_class_id = logits.argmax().item()
			
			print("Stance prediction: {}\n".format(label_list[predicted_class_id]))

	elif data_args.demo_mode == "sample":
		print("\nSampling correct and wrong predictions from validation set...")

		## Build datasets ##
		is_regression, raw_datasets, \
		train_dataset, eval_dataset, predict_dataset = build_datasets(data_args, model_args, training_args, 
																	  config, tokenizer, model)

		## Build trainer ##
		trainer = build_trainer(
			is_regression, data_args, training_args, 
			train_dataset, eval_dataset, 
			tokenizer, model
		)

		## Evaluation
		predict_process(data_args, training_args, eval_dataset, trainer)
	else:
		raise ValueError("demo_mode not specified!")

if __name__ == "__main__":
	main()