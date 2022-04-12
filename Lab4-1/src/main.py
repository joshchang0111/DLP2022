import os
import ipdb
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import random

## Self-defined
from data.dataloader import *
from models.build_model import build_model
from pipelines.trainer import build_trainer

def parse_args():
	parser = argparse.ArgumentParser(description="DLP2022-Lab4-1: EEG Classification")

	## Training options
	parser.add_argument("--train_epoch", type=int, default=300)
	parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
	parser.add_argument("--bs", type=int, default=64, help="Batch size during training")
	parser.add_argument("--activation", type=str, default="relu,leaky_relu,elu")
	parser.add_argument("--dropout", type=float, default=None)
	parser.add_argument("--model", type=str, default="EEGNet", choices=["EEGNet", "DeepConvNet"])
	parser.add_argument("--report_every", type=int, default=10, help="Display results every 10 epochs.")
	parser.add_argument("--seed", type=int, default=None, help="Whether to fix random seed or not.")

	## Others
	parser.add_argument("--save_plot", action="store_true", help="Whether to save the plot or not")
	parser.add_argument("--data_path", type=str, default="../dataset")
	parser.add_argument("--result_path", type=str, default="../result")
	parser.add_argument("--checkpoint_path", type=str, default="../checkpoints")

	args = parser.parse_args()

	return args

if __name__ == "__main__":
	args = parse_args()

	## Set device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Device: {}\n".format(device))

	if device.type != "cuda":
		print("Not using GPU for training!")
		raise SystemExit

	## Set random seed (12 for EEGNet/lr1e-4)
	if args.seed is not None:
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		np.random.seed(seed)
		random.seed(seed)
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True

	## Save experiment results to "best.txt"
	if not os.path.isfile("{}/best.txt".format(args.result_path)):
		with open("{}/best.txt".format(args.result_path), "w") as fw:
			fw.write(
				"{:11s}\t{}\t{}\t{}\t{}\n".format(
					"Model", "activation", "Learning Rate", "Best Epoch", "Test Acc."
				)
			)

	## Load data
	train_data, train_label, test_data, test_label = read_bci_data(args)
	train_loader, test_loader = create_dataset(
		args, device, 
		train_data, train_label, test_data, test_label
	)

	plt.figure()
	activations = args.activation.split(",")
	for activation in activations:
		args.activation = activation
		print("Using {} as activation function...".format(args.activation))
	
		## Build model
		model = build_model(args)

		## Build trainer
		trainer = build_trainer(args, model, device)
		best, epochs, train_accs, test_accs = trainer.train(train_loader, test_loader)

		## Save results
		with open("{}/best.txt".format(args.result_path), "a") as fw:
			#fw.write(
			#	"{:11s}\t{:10s}\t{:13.1E}\t{:10d}\t{:9.4f}\n".format(
			#		args.model, args.activation, args.lr, best["epoch"], best["test_acc"]
			#	)
			#)
			## For dropout
			fw.write(
				"{:11s}\t{:10s}\t{:13.1E}\t{:7.4f}\t{:10d}\t{:9.4f}\n".format(
					args.model, args.activation, args.lr, model.dropout_p, best["epoch"], best["test_acc"]
				)
			)

		plt.plot(epochs, train_accs, linewidth=1, label="{}_train".format(args.activation))
		plt.plot(epochs, test_accs , linewidth=1, label="{}_test".format(args.activation))

	if args.save_plot:
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy (%)")
		plt.title("Activation function comparison ({})".format(args.model))
		plt.legend()
		plt.savefig("{}/{}_lr{:.1E}.png".format(args.result_path, args.model, args.lr))


