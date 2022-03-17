import ipdb
import numpy as np

class FullyConnected:
	"""Fully connected network layer"""
	def __init__(self, in_features, out_features):
		## Parameter size
		self.in_features  = in_features
		self.out_features = out_features

		## Model weight
		self.weight = np.random.randn(self.in_features, self.out_features)
		self.bias   = np.random.randn(1, self.out_features)

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