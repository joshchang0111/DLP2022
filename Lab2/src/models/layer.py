import numpy as np

class FullyConnected:
	"""Fully connected network layer"""
	def __init__(self, in_features, out_features, activation, data_type):
		## Parameter size
		self.in_features  = in_features
		self.out_features = out_features

		## Model weight
		self.weight = np.random.randn(self.in_features, self.out_features)
		self.bias   = np.random.randn(1, self.out_features)

		if activation.lower() == "none" and data_type.lower() == "linear":
			self.weight *= 0.01
			self.bias   *= 0.01
		elif activation.lower() == "relu" and data_type.lower() == "xor":
			self.weight *= 0.1
			self.bias   *= 0.1

		## Gradients
		self.grad_weight = np.zeros((self.in_features, self.out_features))
		self.grad_bias   = np.zeros((1, self.out_features))

	def forward(self, x):
		self.x_for_grad = x ## Save for backward phase
		return x.dot(self.weight) + self.bias

	def backward(self, input_grad):
		self.grad_weight = (self.x_for_grad.T.dot(input_grad))
		self.grad_bias   = input_grad.sum(axis=0, keepdims=True)
		return input_grad.dot(self.weight.T)

	def update(self, lr):
		"""Update weight & bias"""
		self.weight -= lr * self.grad_weight
		self.bias   -= lr * self.grad_bias

class Activation:
	"""Activation function layer"""
	def __init__(self, type_):
		self.type = type_.lower()

	"""
	def forward(self, x):
		if self.type == "sigmoid":
			return 1.0 / (1.0 + np.exp(-x))
		elif self.type == "relu":
			return np.maximum(x, 0)
		else:
			return x

	def backward(self, y):
		if self.type == "sigmoid":
			return np.multiply(y, 1.0 - y)
		elif self.type == "relu":
			return y
		else:
			return 1
	"""
	def forward(self, x):
		if self.type == "sigmoid":
			self.save_for_backward = 1.0 / (1.0 + np.exp(-x))
			return self.save_for_backward
		elif self.type == "relu":
			self.save_for_backward = x
			return np.maximum(x, 0)
		else:
			return x

	def backward(self, input_grad):
		if self.type == "sigmoid":
			current_grad = np.multiply(self.save_for_backward, 1.0 - self.save_for_backward)
			return current_grad * input_grad
		elif self.type == "relu":
			output_grad = input_grad.copy()
			output_grad[self.save_for_backward < 0] = 0 ## gradient where original input < 0 will be 0
			return output_grad
		else:
			return input_grad

