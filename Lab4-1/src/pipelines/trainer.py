import ipdb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

def build_trainer(args, model, device):
	print("Building trainer...\n")
	trainer = Trainer(args, model, device)
	return trainer

class Trainer:
	def __init__(self, args, model, device):
		self.args = args
		self.model = model
		self.device = device

		self.model.to(self.device)
		ipdb.set_trace()

	def train(self, train_loader, test_loader):
		"""Train the model"""
		epochs, train_accs, test_accs = [], [], []
		best = {"epoch": 0, "test_acc": 0}

		loss_fct = nn.CrossEntropyLoss()
		optimizer = optim.Adam(self.model.parameters())

		print("Start training...")
		for epoch in range(self.args.train_epoch):
			## Statistics
			correct, total_samples, total_loss = 0, 0, 0

			self.model.train()
			for inputs, labels in train_loader:
				## Clear the previous gradients in model.
				## Same as optimizer.zero_grad() if using optim.Adam(model.parameters())
				self.model.zero_grad()

				## Forward
				pred_logits = self.model(inputs)
				preds = torch.argmax(pred_logits, dim=1)
				loss = loss_fct(pred_logits, labels)

				## Backward
				loss.backward()
				optimizer.step()

				## Update statistics
				correct += torch.sum(preds == labels).item()
				total_samples += preds.shape[0]
				total_loss += loss.item()

			train_acc = correct / total_samples
			test_acc  = self.test(test_loader)

			## Display statistics
			if epoch % self.args.report_every == 0:
				print(
					"Epoch: {:4d}, Train Loss: {:.4f}, Train Accuracy: {:.4f}, Test Accuracy: {:.4f}".format(
						epoch, total_loss / len(train_loader), train_acc, test_acc
					)
				)

			## Update statistics
			epochs.append(epoch)
			train_accs.append(train_acc)
			test_accs.append(test_acc)
			if test_acc > best["test_acc"]:
				best["epoch"] = epoch
				best["test_acc"] = test_acc

		print("Best Epoch: {}, Test Accuracy: {:.4f}\n".format(best["epoch"], best["test_acc"]))

		return best, epochs, train_accs, test_accs

	def test(self, test_loader):
		"""Test the model every epoch during training."""
		correct, total_samples = 0, 0

		self.model.eval()
		for inputs, labels in test_loader:
			pred_logits = self.model(inputs)
			preds = torch.argmax(pred_logits, dim=1)

			correct += torch.sum(preds == labels).item()
			total_samples += preds.shape[0]

		test_acc = correct / total_samples
		return test_acc
