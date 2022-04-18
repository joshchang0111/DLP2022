import numpy as np
import argparse

## Self-defined
from data.loader import *
from models.model import *
from models.trainer import *
from others.reporter import *

def parse_args():
	parser = argparse.ArgumentParser(description="Lab2: Back Propagation")

	## Training options
	parser.add_argument("-exp_title", action="store_true")
	parser.add_argument("-exp_name", type=str, default="")
	parser.add_argument("-iterations", type=int, default=1)
	parser.add_argument("-train_epoch", type=int, default=100000)
	parser.add_argument("-lr", type=float, default=1e-2, help="Learning rate")
	parser.add_argument("-optim", type=str, default="sgd", choices=["sgd", "adam"])
	parser.add_argument("-report_every", type=int, default=5000)

	parser.add_argument("-activation", type=str, default="sigmoid", choices=["sigmoid", "relu", "none"])
	parser.add_argument("-hidden_dim", type=int, default=50)

	## Others
	parser.add_argument("-data_type", type=str, default="linear", choices=["linear", "xor"])
	parser.add_argument("-result_path", type=str, default="../result")
	parser.add_argument("-checkpoint_path", type=str, default="../checkpoints")

	args = parser.parse_args()

	return args

if __name__ == "__main__":
	args = parse_args()

	## Save experiment results to file
	if args.exp_name != "":
		args.iterations = 10
		if args.exp_title:
			fw = open("{}/exp.txt".format(args.result_path), "a")
			fw.write("\nExperiment: {}\n".format(args.exp_name))
			fw.write("Dataset\tlr   \thidden units\tactivation\tbest epoch\taccuracy\n")
			fw.close()

	## Fix random seed
	np.random.seed(1)

	## Generate input data & label
	x, y = generate_dataset(args.data_type)

	for iter_ in range(args.iterations):
		## Build classifier model
		model = NNClassifier(
			in_features=x.shape[1], 
			out_features=1, 
			hidden_dim=args.hidden_dim, 
			activation=args.activation, 
			data_type=args.data_type
		)

		## Start training
		pred_y, epochs, losses, acc = train(args, x, y, model)

		## Save results
		if args.exp_name != "":
			show_experiment(args, epochs, acc)
		else:
			show_result(args, x, y, pred_y)
			plot_learning_curve(args, epochs, losses)
