import pdb
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim

def build_trainer(args, model, device):
	print("Build trainer...")
	trainer = Trainer(args, model, device)
	return trainer

class Trainer:
	def __init__(self, args, model, device):
		self.args = args
		self.model = model
		self.device = device

		self.model.to(self.device)

		## Load model checkpoint if test only
		if self.args.test:
			pretrained_str = "w" if self.args.pretrained else "wo"
			model_path = "{}/{}_{}_pretrained.pt".format(self.args.checkpoint_path, self.args.model, pretrained_str)

			print("Loading model checkpoint from {}\n".format(model_path))
			self.model.load_state_dict(
				torch.load(model_path)
			)

	def train(self, train_loader, test_loader, loss_weight):
		"""Train the model"""
		epochs, train_accs, test_accs = [], [], []
		best = {"test_acc": 0}

		if loss_weight is not None:
			loss_weight = torch.Tensor(loss_weight).to(self.device)
		loss_fct = nn.CrossEntropyLoss(weight=loss_weight)

		if self.args.optim == "sgd":
			optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
		elif self.args.optim == "adam":
			optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, betas=[self.args.beta1, self.args.beta2])

		for epoch in range(self.args.train_epoch):
			all_labels, all_preds, total_loss = [], [], 0
			
			self.model.train()
			for inputs, labels in tqdm(train_loader):
				## Move to device (GPU)
				inputs = inputs.to(self.device)
				labels = labels.to(self.device)

				## Forward
				self.model.zero_grad()
				logits = self.model(inputs)
				preds = torch.argmax(logits, dim=1)
				loss = loss_fct(logits, labels)

				## Backward
				loss.backward()
				optimizer.step()

				## Update Statistics
				all_labels.extend(labels.tolist())
				all_preds.extend(preds.tolist())
				total_loss += loss.item()

			all_labels = torch.Tensor(all_labels)
			all_preds  = torch.Tensor(all_preds)
			train_acc = torch.sum(all_preds == all_labels).item() / len(all_labels)
			test_acc, test_f1s, test_f1_macro, test_labels, test_preds = self.test(test_loader)

			## Display statistics
			if epoch % self.args.report_every == 0:
				print(
					("Epoch: {:2d}/{:2d}, Train Loss: {:.4f}, Train Accuracy: {:.4f}, Test Accuracy: {:.4f}\n" + 
					 "F1_0: {:.4f}, F1_1: {:.4f}, F1_2: {:.4f}, F1_3: {:.4f}, F1_4: {:.4f}, F1_Macro: {:.4f}").format(
						epoch, self.args.train_epoch, total_loss / len(train_loader), train_acc, test_acc, 
						test_f1s[0], test_f1s[1], test_f1s[2], test_f1s[3], test_f1s[4], test_f1_macro
					)
				)

			## Update statistics
			epochs.append(epoch)
			train_accs.append(train_acc)
			test_accs.append(test_acc)
			if test_acc > best["test_acc"]:
				print("Saving model & results with best test accuracy...\n")
				best["epoch"] = epoch
				best["test_acc"] = test_acc
				best["f1s"] = test_f1s
				best["f1_macro"] = test_f1_macro

				## For confusion matrix
				best["labels"] = test_labels
				best["preds"]  = test_preds

				## TODO: save model checkpoint
				pretrained_str = "w" if self.args.pretrained else "wo"
				checkpoint_file = "{}/{}_{}_pretrained.pt".format(self.args.checkpoint_path, self.args.model, pretrained_str)
				torch.save(self.model.state_dict(), checkpoint_file)

		print("Best Epoch: {}, Test Accuracy: {:.4f}, Test F1_Macro: {:.4f}\n".format(best["epoch"], best["test_acc"], best["f1_macro"]))

		return best, epochs, train_accs, test_accs

	def test(self, test_loader):
		"""Test the model every epoch during training."""
		all_labels, all_preds = [], []

		self.model.eval()
		with torch.no_grad(): ## Important: or gpu will go out of memory
			for idx, (inputs, labels) in enumerate(tqdm(test_loader)):
				## Move to device (GPU)
				inputs = inputs.to(self.device)
				labels = labels.to(self.device)
				
				logits = self.model(inputs)
				preds = torch.argmax(logits, dim=1)

				## For later evaluation
				all_labels.extend(labels.tolist())
				all_preds.extend(preds.tolist())

		## Evaluation metrics
		all_labels = torch.Tensor(all_labels).to(torch.long)
		all_preds  = torch.Tensor(all_preds).to(torch.long)
		test_acc = torch.sum(all_preds == all_labels).item() / len(all_labels)
		test_f1s = f1_score(all_labels, all_preds, average=None)
		test_f1_macro = f1_score(all_labels, all_preds, average="macro")

		return test_acc, test_f1s, test_f1_macro, all_labels, all_preds


