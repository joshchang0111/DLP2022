import os
import ipdb
import logging
import numpy as np

## Get the same logger as in `main.py`
logger = logging.getLogger("__main__")

def train_process(data_args, training_args, last_checkpoint, train_dataset, trainer):
	checkpoint = None
	if training_args.resume_from_checkpoint is not None:
		checkpoint = training_args.resume_from_checkpoint
	elif last_checkpoint is not None:
		checkpoint = last_checkpoint
	train_result = trainer.train(resume_from_checkpoint=checkpoint)
	metrics = train_result.metrics
	max_train_samples = (
		data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
	)
	metrics["train_samples"] = min(max_train_samples, len(train_dataset))

	#trainer.save_model()  # Saves the tokenizer too for easy upload

	trainer.log_metrics("train", metrics)
	#trainer.save_metrics("train", metrics)
	#trainer.save_state()

	return trainer.best_checkpoint_path

def eval_process(data_args, training_args, raw_datasets, eval_dataset, trainer):
	logger.info("*** Evaluate ***")

	## Loop to handle MNLI double evaluation (matched, mis-matched)
	tasks = [data_args.task_name]
	eval_datasets = [eval_dataset]
	if data_args.task_name == "mnli":
		tasks.append("mnli-mm")
		eval_datasets.append(raw_datasets["validation_mismatched"])
		combined = {}

	for eval_dataset, task in zip(eval_datasets, tasks):
		metrics = trainer.evaluate(eval_dataset=eval_dataset)

		max_eval_samples = (
			data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
		)
		metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

		if task == "mnli-mm":
			metrics = {k + "_mm": v for k, v in metrics.items()}
		if task is not None and "mnli" in task:
			combined.update(metrics)

		trainer.log_metrics("eval", metrics)
		trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

		## Write overall results (model, acc, f1-macro, ...)
		if not os.path.isfile(training_args.overall_results_path):
			with open(training_args.overall_results_path, "w") as fw:
				f1s = ["{:10s}".format("F1-{}".format(label_i)) for label_i in range(data_args.num_labels)]
				metrics2report = [
					"{:20s}".format("Model"), 
					"{:10s}".format("Fold-Type"),
					"{:4s}".format("Fold"),
					"{:10s}".format("Acc"),
					"{:10s}".format("F1-Macro")
				]
				metrics2report.extend(f1s)
				fw.write("{}\n".format("\t".join(metrics2report)))

		with open(training_args.overall_results_path, "a") as fw:
			f1s = ["{:<10.4f}".format(metrics["eval_f1_{}".format(label_i)]) for label_i in range(data_args.num_labels)]
			metrics2report = [
				"{:20s}".format(data_args.model_name[:20]),
				"{:10s}".format("Target" if data_args.target_wise else "5-Fold"),
				"{:4d}".format(data_args.fold), 
				"{:<10.4f}".format(metrics["eval_accuracy"]), 
				"{:<10.4f}".format(metrics["eval_f1_macro"])
			]
			metrics2report.extend(f1s)
			fw.write("{}\n".format("\t".join(metrics2report)))

def predict_process(data_args, training_args, predict_dataset, trainer):
	'''Only used when demo'''
	def display_sample(idx):
		label_list = data_args.label_list.split(" ")

		print("Target: {}".format(predict_dataset["target"][idx]))
		print("Claim : {}".format(predict_dataset["claim"][idx]))
		print("Prediction: {}".format(label_list[predictions[idx]]))
		print("GT-Label  : {}\n".format(label_list[predict_dataset["label"][idx]]))

	logger.info("*** Predict ***")

	# Removing the `label` columns because it contains -1 and Trainer won't like that.
	pred_result = trainer.predict(predict_dataset, metric_key_prefix="predict")
	predictions = pred_result.predictions
	predictions = np.argmax(predictions, axis=1)

	## Display metrics
	print()
	print("#############")
	print("## Metrics ##")
	print("#############")
	print("Accuracy: {:.4f}".format(pred_result.metrics["predict_accuracy"]))
	print("F1-Macro: {:.4f}".format(pred_result.metrics["predict_f1_macro"]))
	print("F1-0    : {:.4f}".format(pred_result.metrics["predict_f1_0"]))
	print("F1-1    : {:.4f}".format(pred_result.metrics["predict_f1_1"]))
	if "predict_f1_2" in pred_result.metrics:
		print("F1-2    : {:.4f}".format(pred_result.metrics["predict_f1_2"]))

	print("########################")
	print("## Correct Prediction ##")
	print("########################")

	## Find correct predictions for each class
	for class_i in list(set(pred_result.label_ids)):
		correct_idx = np.where((predictions == pred_result.label_ids) & (pred_result.label_ids == class_i))[0]
		display_sample(correct_idx[0])

	print("######################")
	print("## Wrong Prediction ##")
	print("######################")

	for class_i in list(set(pred_result.label_ids)):
		wrong_idx = np.where((predictions != pred_result.label_ids) & (pred_result.label_ids == class_i))[0]
		display_sample(wrong_idx[0])



