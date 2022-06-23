"""
Finetuning the library models for sequence classification on GLUE.
This code is developed based on:
	https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
"""
import os
import sys
import ipdb
import logging

import datasets
import numpy as np

import transformers
from transformers import (
	HfArgumentParser,
	TrainingArguments,
	set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

## Self-defined
from others.args import DataTrainingArguments, ModelArguments, CustomTrainingArguments
from others.processes import train_process, eval_process, predict_process
from data.build_datasets import build_datasets
from models.load_model import load_model
from pipelines.build_trainer import build_trainer

## Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.19.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)

def main():
	## See all possible arguments in src/transformers/training_args.py
	## or by passing the --help flag to this script.
	## We now keep distinct sets of args, for a cleaner separation of concerns.

	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
	if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
		## If we pass only one argument to the script and it's the path to a json file,
		## let's parse it to get our arguments.
		model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
	else:
		model_args, data_args, training_args = parser.parse_args_into_dataclasses()

	## Setup some paths
	training_args.output_dir = "{}/{}/{}".format(training_args.output_dir, data_args.dataset_name, model_args.model_name_or_path.replace("/", "."))
	training_args.overall_results_path = "{}/results.txt".format(training_args.output_dir)
	if data_args.target_wise:
		training_args.output_dir = "{}/target_{}".format(training_args.output_dir, data_args.fold)
	else:
		training_args.output_dir = "{}/split_{}".format(training_args.output_dir, data_args.fold)
	data_args.model_name = model_args.model_name_or_path
	os.makedirs(training_args.output_dir, exist_ok=True)

	## Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		handlers=[logging.StreamHandler(sys.stdout)],
	)

	#log_level = training_args.get_process_log_level()
	log_level = logging.WARNING ## only report errors & warnings
	logger.setLevel(log_level)
	datasets.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.enable_default_handler()
	transformers.utils.logging.enable_explicit_format()

	## Log on each process the small summary:
	logger.warning(
		f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
		+ f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
	)
	logger.info(f"Training/evaluation parameters {training_args}")

	## Detecting last checkpoint.
	last_checkpoint = None
	if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
		last_checkpoint = get_last_checkpoint(training_args.output_dir)
		if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
			raise ValueError(
				f"Output directory ({training_args.output_dir}) already exists and is not empty. "
				"Use --overwrite_output_dir to overcome."
			)
		elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
			logger.info(
				f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
				"the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
			)

	## Set seed before initializing model.
	set_seed(training_args.seed)

	################
	## Load model ##
	################
	config, tokenizer, model = load_model(data_args, model_args, training_args)

	####################
	## Build datasets ##
	####################
	is_regression, raw_datasets, \
	train_dataset, eval_dataset, predict_dataset = build_datasets(data_args, model_args, training_args, 
																  config, tokenizer, model)

	###################
	## Build trainer ##
	###################
	trainer = build_trainer(
		is_regression, data_args, training_args, 
		train_dataset, eval_dataset, 
		tokenizer, model
	)

	## Training
	if training_args.do_train:
		training_args.best_checkpoint_path = train_process(data_args, training_args, last_checkpoint, train_dataset, trainer)

	## Evaluation
	if training_args.do_eval:
		## Load new model and build new trainer if doing final evaluation after training
		if training_args.do_train:
			config, tokenizer, model = load_model(data_args, model_args, training_args)
			trainer = build_trainer(
				is_regression, data_args, training_args, 
				train_dataset, eval_dataset, 
				tokenizer, model
			)
		eval_process(data_args, training_args, raw_datasets, eval_dataset, trainer)

	## Prediction
	if training_args.do_predict:
		predict_process(is_regression, data_args, training_args, raw_datasets, predict_dataset, trainer)

if __name__ == "__main__":
	main()