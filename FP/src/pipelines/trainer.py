"""
This code is developed based on:
	https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
"""
import os
import ipdb
import time
import shutil
from typing import Optional, Union, Dict, Any, Callable, Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from transformers.utils import logging

from others.args import DataTrainingArguments, ModelArguments

logger = logging.get_logger(__name__)

class CustomTrainer(Trainer):
	def __init__(
		self,
		model: Union[PreTrainedModel, nn.Module] = None,
		args: TrainingArguments = None, 
		model_args: ModelArguments = None, 
		data_args: DataTrainingArguments = None, 
		data_collator: Optional[DataCollator] = None,
		train_dataset: Optional[Dataset] = None,
		eval_dataset: Optional[Dataset] = None,
		tokenizer: Optional[PreTrainedTokenizerBase] = None,
		model_init: Callable[[], PreTrainedModel] = None,
		compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
		callbacks: Optional[List[TrainerCallback]] = None,
		optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
		preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
	):
		Trainer.__init__(
			self, 
			model, 
			args, 
			data_collator, 
			train_dataset, 
			eval_dataset, 
			tokenizer, 
			model_init, 
			compute_metrics, 
			callbacks, 
			optimizers, 
			preprocess_logits_for_metrics
		)

		self.data_args  = data_args
		self.model_args = model_args

		## For best model saving
		self._ckpt_eval_loss = {}
		if self.args.save_model_accord_to_metric:
			self._ckpt_eval_metric = {}

		self.best_checkpoint_path = None

		## NEW: compute loss weight based on train_dataset
		if self.args.do_train:
			labels_tensor = torch.Tensor(train_dataset["label"])
			n_each_label  = torch.Tensor(
				[(labels_tensor == i).sum() for i in range(len(set(train_dataset["label"])))]
			)
			#self.loss_weight = (n_each_label.max() / n_each_label).to(self.model.device)
		self.loss_weight = None

	def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
		"""
		Modification:
			- record current eval loss / metric for best model saving
		"""
		if self.control.should_log:
			#if is_torch_tpu_available():
			#	xm.mark_step()

			logs: Dict[str, float] = {}

			# all_gather + mean() to get average loss over all processes
			tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

			# reset tr_loss to zero
			tr_loss -= tr_loss

			logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
			logs["learning_rate"] = self._get_learning_rate()

			self._total_loss_scalar += tr_loss_scalar
			self._globalstep_last_logged = self.state.global_step
			self.store_flos()

			self.log(logs)

		metrics = None
		if self.control.should_evaluate:
			metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
			self._report_to_hp_search(trial, epoch, metrics)

			## NEW: record metric
			print("\n*** Evaluation Results ***")
			print("Loss: {:.4f}, Accuracy: {:.4f}".format(metrics["eval_loss"], metrics["eval_accuracy"]))
			f1s = ["F1_{}: {:.4f}".format(key.split("_")[-1].capitalize(), metrics[key]) for key in metrics.keys() if "f1" in key]
			print(", ".join(f1s))
			print()

			if self.args.save_model_accord_to_metric:
				self._cur_eval_metric = metrics["eval_f1_macro"] if "eval_f1_macro" in metrics else metrics["eval_accuracy"]
			self._cur_eval_loss = metrics["eval_loss"]

		if self.control.should_save:
			self._save_checkpoint(model, trial, metrics=metrics)
			self.control = self.callback_handler.on_save(self.args, self.state, self.control)

	def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
		"""
		Modification:
			- record eval loss / metric and maintain best model
		"""
		## NEW
		if self.args.save_strategy == "steps":
			if self.args.eval_steps != self.args.save_steps:
				raise Exception(
					"To properly store best models, please make sure eval_steps equals to save_steps."
				)

		if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
			return

		# Check if we should delete older checkpoint(s)
		checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)

		## NEW: record the eval metric for the last checkpoint
		self._ckpt_eval_loss[checkpoints_sorted[-1]] = self._cur_eval_loss
		if self.args.save_model_accord_to_metric:
			self._ckpt_eval_metric[checkpoints_sorted[-1]] = self._cur_eval_metric

		if len(checkpoints_sorted) <= self.args.save_total_limit:
			return

		number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)

		## NEW: sort checkpoints path
		if self.args.save_model_accord_to_metric:
			## sort according to metric (ascending for metric)
			checkpoints_sorted = [
				k for k, v in sorted(self._ckpt_eval_metric.items(),
									 key=lambda x: x[1],
									 reverse=False)
			]
			checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
		else:
			## sort according to loss (descending for loss)
			checkpoints_sorted = [
				k for k, v in sorted(self._ckpt_eval_loss.items(),
									 key=lambda x: x[1],
									 reverse=True)
			]

		#checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
		for checkpoint in checkpoints_to_be_deleted:
			logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
			shutil.rmtree(checkpoint)

			## NEW: remove the deleted ckpt
			del self._ckpt_eval_loss[checkpoint]
			if self.args.save_model_accord_to_metric:
				del self._ckpt_eval_metric[checkpoint]

		self.best_checkpoint_path = checkpoints_sorted[-1]

	def _save(self, output_dir: Optional[str] = None, state_dict=None):
		"""
		Modification:
			- Also record model_args and data_args
		"""
		# If we are executing this function, we are the process zero, so we don't check for that.
		output_dir = output_dir if output_dir is not None else self.args.output_dir
		os.makedirs(output_dir, exist_ok=True)
		logger.info(f"Saving model checkpoint to {output_dir}")
		# Save a trained model and configuration using `save_pretrained()`.
		# They can then be reloaded using `from_pretrained()`
		if not isinstance(self.model, PreTrainedModel):
			if isinstance(unwrap_model(self.model), PreTrainedModel):
				if state_dict is None:
					state_dict = self.model.state_dict()
				unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
			else:
				logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
				if state_dict is None:
					state_dict = self.model.state_dict()
				torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
		else:
			self.model.save_pretrained(output_dir, state_dict=state_dict)
		if self.tokenizer is not None:
			self.tokenizer.save_pretrained(output_dir)

		# Good practice: save your training arguments together with the trained model
		torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
		## New
		torch.save(self.data_args , os.path.join(output_dir, "data_args.bin"))
		torch.save(self.model_args, os.path.join(output_dir, "model_args.bin"))

	def compute_loss(self, model, inputs, return_outputs=False):
		"""
		Modification:
			- enable weighted-loss during training
		"""
		labels = inputs.get("labels")
		
		## Forward pass
		outputs = model(**inputs)
		logits = outputs.get("logits")
		
		## Compute customized loss value
		loss_fct = nn.CrossEntropyLoss(weight=self.loss_weight)
		loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

		return (loss, outputs) if return_outputs else loss

		"""
		if self.label_smoother is not None and "labels" in inputs:
			labels = inputs.pop("labels")
		else:
			labels = None
		outputs = model(**inputs)
		# Save past state if it exists
		# TODO: this needs to be fixed and made cleaner later.
		if self.args.past_index >= 0:
			self._past = outputs[self.args.past_index]

		if labels is not None:
			loss = self.label_smoother(outputs, labels)
		else:
			# We don't use .loss here since the model may return tuples instead of ModelOutput.
			loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

		return (loss, outputs) if return_outputs else loss
		"""
