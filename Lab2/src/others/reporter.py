import numpy as np
import matplotlib.pyplot as plt

def show_result(args, x, y, pred_y):
	"""
	Visualization of input and predicted results.
	Parameters:
		args: arguments from argparse
		x: inputs (2-dimensional array)
		y: ground truth label (1-dimensional array)
		pred_y: outputs of neural network (1-dimensional array)
	"""
	plt.figure(figsize=(10, 5))
	plt.subplot(1, 2, 1)
	plt.title("Ground truth", fontsize=18)
	for i in range(x.shape[0]):
		if y[i] == 0:
			plt.plot(x[i][0], x[i][1], "ro")
		else:
			plt.plot(x[i][0], x[i][1], "bo")

	plt.subplot(1, 2, 2)
	plt.title("Predict result", fontsize=18)
	for i in range(x.shape[0]):
		if pred_y[i] == 0:
			plt.plot(x[i][0], x[i][1], "ro")
		else:
			plt.plot(x[i][0], x[i][1], "bo")

	plt.savefig("{}/{}_result.png".format(args.result_path, args.data_type))

def plot_learning_curve(args, epochs, losses):
	plt.figure()
	plt.plot(epochs, losses)
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("{}".format(args.data_type))
	plt.savefig("{}/{}_loss.png".format(args.result_path, args.data_type))

def show_experiment(args, epochs, acc):
	if args.exp_name != "":
		fw = open("{}/exp.txt".format(args.result_path), "a")
		fw.write(
			"{:7s}\t{:5.4f}\t{:12d}\t{:10s}\t{:10d}\t{:8.4f}\n".format(
				args.data_type, 
				args.lr, 
				args.hidden_dim, 
				args.activation, 
				epochs[-1], 
				acc
			)
		)
		fw.close()

def plot_sigmoid():
	x = np.linspace(-10, 10, 100)
	y = 1 / (1 + np.exp(-x))
	z = np.multiply(y, 1.0 - y)
  
	plt.plot(x, y, label="sigmoid")
	plt.plot(x, z, label="sigmoid derivative")
	plt.xlabel("x")
	#plt.ylabel("Sigmoid")
	plt.legend()

	plt.savefig("../result/sigmoid.png")

if __name__ == "__main__":
	plot_sigmoid()