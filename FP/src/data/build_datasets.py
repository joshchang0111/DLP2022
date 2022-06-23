import ipdb
import random
import logging

from datasets import load_dataset
from transformers import PretrainedConfig

## Self-defined
from others.args import task_to_keys

logger = logging.getLogger("__main__")

def build_datasets(data_args, model_args, training_args, config, tokenizer, model):
	print("\nBuilding datasets...")
	##################
	## Load Dataset ##
	##################
	# Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
	# or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
	#
	# For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
	# sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
	# label if at least two columns are provided.
	#
	# If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
	# single column. You can easily tweak this behavior (see below)
	#
	# In distributed training, the load_dataset function guarantee that only one local process can concurrently
	# download the dataset.
	if data_args.task_name is not None:
		# Downloading and loading a dataset from the hub.
		raw_datasets = load_dataset(
			"glue",
			data_args.task_name,
			cache_dir=model_args.cache_dir,
			use_auth_token=True if model_args.use_auth_token else None,
		)
	elif data_args.load_dataset_from_hub: #data_args.dataset_name is not None:
		# Downloading and loading a dataset from the hub.
		raw_datasets = load_dataset(
			data_args.dataset_name,
			data_args.dataset_config_name,
			cache_dir=model_args.cache_dir,
			use_auth_token=True if model_args.use_auth_token else None,
		)
	else:
		## Loading a dataset from your local files.
		## CSV/JSON training and evaluation files are needed.
		if data_args.target_wise:
			print("Loading target [{}]...".format(data_args.fold))
			data_files = {
				"train"     : "{}/{}/target_{}/{}".format(data_args.dataset_root, data_args.dataset_name, data_args.fold, data_args.train_file), 
				"validation": "{}/{}/target_{}/{}".format(data_args.dataset_root, data_args.dataset_name, data_args.fold, data_args.validation_file)
			}
		else:
			print("Loading split [{}]".format(data_args.fold))
			data_files = {
				"train"     : "{}/{}/split_{}/{}".format(data_args.dataset_root, data_args.dataset_name, data_args.fold, data_args.train_file), 
				"validation": "{}/{}/split_{}/{}".format(data_args.dataset_root, data_args.dataset_name, data_args.fold, data_args.validation_file)
			}

		## Get the test dataset: you can provide your own CSV/JSON test file (see below)
		## when you use `do_predict` without specifying a GLUE benchmark task.
		if training_args.do_predict:
			if data_args.test_file is not None:
				train_extension = data_args.train_file.split(".")[-1]
				test_extension = data_args.test_file.split(".")[-1]
				assert (
					test_extension == train_extension
				), "`test_file` should have the same extension (csv or json) as `train_file`."
				data_files["test"] = data_args.test_file
			else:
				raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

		for key in data_files.keys():
			logger.info(f"load a local file for {key}: {data_files[key]}")

		if data_args.train_file.endswith(".csv"):
			## Loading a dataset from local csv files
			raw_datasets = load_dataset(
				"csv",
				data_files=data_files,
				cache_dir=model_args.cache_dir,
				use_auth_token=True if model_args.use_auth_token else None,
			)
		else:
			# Loading a dataset from local json files
			raw_datasets = load_dataset(
				"json",
				data_files=data_files,
				cache_dir=model_args.cache_dir,
				use_auth_token=True if model_args.use_auth_token else None,
			)

	# See more about loading any type of standard or custom dataset at
	# https://huggingface.co/docs/datasets/loading_datasets.html.

	################
	## Preprocess ##
	################
	## Labels
	if data_args.task_name is not None:
		is_regression = data_args.task_name == "stsb"
		if not is_regression:
			label_list = raw_datasets["train"].features["label"].names
			num_labels = len(label_list)
		else:
			num_labels = 1
	else:
		# Trying to have good defaults here, don't hesitate to tweak to your needs.
		#is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
		is_regression = len(raw_datasets["train"].unique("stance")) == 1
		if is_regression:
			num_labels = 1
		else:
			# A useful fast method:
			# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
			label_list = raw_datasets["train"].unique("stance")
			label_list.sort()  # Let's sort it for determinism
			num_labels = len(label_list)
			data_args.label_list = " ".join(label_list)

	assert model.config.num_labels == num_labels, "num_labels specified in data_args should be the same as dataset!"

	## Preprocessing the raw_datasets
	if data_args.task_name is not None:
		sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
	else:
		# Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
		"""
		non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
		if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
			sentence1_key, sentence2_key = "sentence1", "sentence2"
		else:
			if len(non_label_column_names) >= 2:
				sentence1_key, sentence2_key = non_label_column_names[:2]
			else:
				sentence1_key, sentence2_key = non_label_column_names[0], None
		"""
		target_key, claim_key = raw_datasets["train"].column_names[0], raw_datasets["train"].column_names[1]

	## Padding strategy
	if data_args.pad_to_max_length: ## default
		padding = "max_length"
	else:
		# We will pad later, dynamically at batch creation, to the max sequence length in each batch
		padding = False

	## Some models have set the order of the labels to use, so let's make sure we do use it.
	label_to_id = None
	if (
		model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
		and data_args.task_name is not None
		and not is_regression
	):
		# Some have all caps in their config, some don't.
		label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
		if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
			label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
		else:
			logger.warning(
				"Your model seems to have been trained with labels, but they don't match the dataset: ",
				f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
				"\nIgnoring the model labels as a result.",
			)
	elif data_args.task_name is None and not is_regression:
		label_to_id = {v: i for i, v in enumerate(label_list)}

	if label_to_id is not None:
		model.config.label2id = label_to_id
		model.config.id2label = {id: label for label, id in config.label2id.items()}
	elif data_args.task_name is not None and not is_regression:
		model.config.label2id = {l: i for i, l in enumerate(label_list)}
		model.config.id2label = {id: label for label, id in config.label2id.items()}

	if data_args.max_seq_length > tokenizer.model_max_length:
		logger.warning(
			f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
			f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
		)
	max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

	def preprocess_function(examples):
		## Tokenize the texts
		#args = (
		#	(examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
		#)
		args = (examples[claim_key], examples[target_key])
		result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

		# Map labels to IDs (not necessary for GLUE tasks)
		if label_to_id is not None and "stance" in examples:
			result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["stance"]]
		return result

	## Preprocessing
	with training_args.main_process_first(desc="dataset map pre-processing"):
		raw_datasets = raw_datasets.map(
			preprocess_function,
			batched=True,
			load_from_cache_file=not data_args.overwrite_cache,
			desc="Running tokenizer on dataset",
		)

	train_dataset, eval_dataset, predict_dataset = None, None, None
	if training_args.do_train:
		if "train" not in raw_datasets:
			raise ValueError("--do_train requires a train dataset")
		train_dataset = raw_datasets["train"]
		if data_args.max_train_samples is not None:
			max_train_samples = min(len(train_dataset), data_args.max_train_samples)
			train_dataset = train_dataset.select(range(max_train_samples))

	if training_args.do_eval:
		if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
			raise ValueError("--do_eval requires a validation dataset")
		eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
		if data_args.max_eval_samples is not None:
			max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
			eval_dataset = eval_dataset.select(range(max_eval_samples))

	if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
		if "test" not in raw_datasets and "test_matched" not in raw_datasets:
			raise ValueError("--do_predict requires a test dataset")
		predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
		if data_args.max_predict_samples is not None:
			max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
			predict_dataset = predict_dataset.select(range(max_predict_samples))

	## Log a few random samples from the training set:
	if training_args.do_train:
		for index in random.sample(range(len(train_dataset)), 3):
			logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

	return is_regression, raw_datasets, train_dataset, eval_dataset, predict_dataset


