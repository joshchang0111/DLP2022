import ipdb
import numpy as np
import argparse

## Self-defined
from data.loader import *
from models.model import *
from models.trainer import *

def parse_args():
	parser = argparse.ArgumentParser(description="Rumor Detection")

	## Training options
	parser.add_argument("-debug", action="store_true")
	parser.add_argument("-train_epoch", type=int, default=100000)
	parser.add_argument("-lr", type=float, default=1e-2, help="Learning rate")
	parser.add_argument("-optim", type=str, default="sgd", choices=["sgd", "adam"])
	parser.add_argument("-report_every", type=int, default=5000)

	parser.add_argument("-hidden_dim", type=int, default=50)

	## Others
	parser.add_argument("-data_type", type=str, default="linear", choices=["linear", "xor"])
	parser.add_argument("-result_path", type=str, default="../result")
	parser.add_argument("-checkpoint_path", type=str, default="../checkpoints")

	args = parser.parse_args()

	return args

if __name__ == "__main__":
	args = parse_args()

	## Fix random seed
	np.random.seed(1)

	## Generate input data & label
	x, y = generate_dataset(args.data_type)

	## Build classifier model
	model = NNClassifier(
		in_features=x.shape[1], 
		out_features=1, 
		hidden_dim=args.hidden_dim
	)

	## Start training
	pred_y = train(args, x, y, model)

	show_result(args, x, y, pred_y)