import os
import ipdb

from transformers import (
	AutoConfig,
	AutoModelForSequenceClassification,
	AutoTokenizer,
	PretrainedConfig
)

def load_model(data_args, model_args, training_args):
	# Load pretrained model and tokenizer
	#
	# In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
	# download model & vocab.
	print("\nLoading pretrained model and tokenizer...")
	print("Language Model: {}".format(model_args.model_name_or_path))

	if training_args.best_checkpoint_path is not None:
		if os.path.isdir(training_args.best_checkpoint_path):
			print()
			model_args.model_name_or_path = training_args.best_checkpoint_path
			model_args.config_name = training_args.best_checkpoint_path
			model_args.tokenizer_name = training_args.best_checkpoint_path
		else:
			raise Exception(
				"Best checkpoint path {} doesn't exist!".format(training_args.best_checkpoint_path)
			)

	config = AutoConfig.from_pretrained(
		model_args.config_name if model_args.config_name else model_args.model_name_or_path,
		num_labels=data_args.num_labels,
		finetuning_task=data_args.task_name,
		cache_dir=model_args.cache_dir,
		revision=model_args.model_revision,
		use_auth_token=True if model_args.use_auth_token else None,
	)
	
	tokenizer = AutoTokenizer.from_pretrained(
		model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
		cache_dir=model_args.cache_dir,
		use_fast=model_args.use_fast_tokenizer,
		revision=model_args.model_revision,
		use_auth_token=True if model_args.use_auth_token else None,
	)
	
	model = AutoModelForSequenceClassification.from_pretrained(
		model_args.model_name_or_path,
		from_tf=bool(".ckpt" in model_args.model_name_or_path),
		config=config,
		cache_dir=model_args.cache_dir,
		revision=model_args.model_revision,
		use_auth_token=True if model_args.use_auth_token else None,
	)

	return config, tokenizer, model