import numpy as np

def loss_fct(preds, y):
	"""Calculate loss value, using mean squared error."""
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
	epochs = []
	losses = []
	for epoch in range(args.train_epoch):
		## Forward phase
		preds = model.forward(x)

		## Calculate loss values & predictions
		loss = loss_fct(preds, y)
		pred_y, acc = accuracy(preds, y)

		## Record
		epochs.append(epoch)
		losses.append(loss)

		## Print result per 5000 epochs
		if epoch % args.report_every == 0:
			print("epoch {:5d}, loss: {:.4f}, accuracy: {:.4f}".format(epoch, loss, acc))

		## Stop criterion
		if acc == 1:
			print("epoch {:5d}, loss: {:.4f}, accuracy: {:.4f}".format(epoch, loss, acc))
			if args.exp_name == "":
				print(preds)
				print()
			break

		## Backward phase
		model.backward(loss_derivative(preds, y))
		model.update(args.lr)

	return pred_y, epochs, losses, acc

