import ipdb
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

def loss_fct(preds, y):
	"""Calculate loss value."""
	return np.mean((preds - y) ** 2)

def loss_derivative(preds, y):
	"""Loss backward phase"""
	return 2 * (preds - y) / y.shape[0]

def accuracy(preds, y):
	"""
	Obtain accuracy.
	Parameters:
		preds: predictions from model directly
		y: ground truth label
	"""
	## Convert probability into class label.
	pred_y = np.zeros(preds.shape)
	pred_y[np.absolute(preds >= 0.5)] = 1
	pred_y[np.absolute(preds <  0.5)] = 0

	## Calculate accuracy
	acc = (pred_y == y).sum() / pred_y.shape[0]
	return pred_y, acc

def train(args, x, y, model):
	"""Main training loops"""
	print("Start training process.")
	for epoch in range(args.train_epoch):
		## Forward phase
		preds = model.forward(x)

		## Calculate loss values & predictions
		loss = loss_fct(preds, y)
		pred_y, acc = accuracy(preds, y)

		## Print result per 5000 epochs
		if epoch % args.report_every == 0:
			print("epoch {:5d}, loss: {:.4f}, accuracy: {:.4f}".format(epoch, loss, acc))

		## Stop criterion
		if acc == 1:
			print("epoch {:5d}, loss: {:.4f}, accuracy: {:.4f}\n".format(epoch, loss, acc))
			break

		## Backward phase
		model.backward(loss_derivative(preds, y))
		model.update(args.lr)

	return pred_y