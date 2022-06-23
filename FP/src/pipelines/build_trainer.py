import ipdb
import numpy as np

from datasets import load_metric
from transformers import (
	Trainer, 
	EvalPrediction, 
	default_data_collator, 
	DataCollatorWithPadding
)

## Self-defined
from pipelines.trainer import CustomTrainer

def build_trainer(
		is_regression, data_args, training_args, 
		train_dataset, eval_dataset, 
		tokenizer, model
	):
	print("\nBuilding trainer...")

	## Get the metric function
	if data_args.task_name is not None:
		metric = load_metric("glue", data_args.task_name)
	else:
		#metric = load_metric("accuracy")
		metric = {
			"accuracy": load_metric("accuracy"), 
			"f1": load_metric("f1")
		}

	## You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
	## predictions and label_ids field) and has to return a dictionary string to float.
	def compute_metrics(p: EvalPrediction):
		preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
		preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
		if data_args.task_name is not None:
			result = metric.compute(predictions=preds, references=p.label_ids)
			if len(result) > 1:
				result["combined_score"] = np.mean(list(result.values())).item()
			return result
		#elif is_regression:
		#	return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
		else:
			#return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
			result = {
				"accuracy": metric["accuracy"].compute(predictions=preds, references=p.label_ids)["accuracy"], 
				"f1_macro": metric["f1"].compute(predictions=preds, references=p.label_ids, average="macro")["f1"]
			}
			for label_i in list(model.config.label2id.values()): #range(data_args.num_labels):
				result["f1_{}".format(label_i)] = metric["f1"].compute(predictions=preds, references=p.label_ids, average=None)["f1"][label_i]
			return result

	## Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
	## we already did the padding.
	if data_args.pad_to_max_length:
		data_collator = default_data_collator
	elif training_args.fp16:
		data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
	else:
		data_collator = None

	## Initialize our Trainer
	trainer = CustomTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset if training_args.do_train else None,
		eval_dataset=eval_dataset if training_args.do_eval else None,
		compute_metrics=compute_metrics,
		tokenizer=tokenizer,
		data_collator=data_collator,
	)

	return trainer

