import numpy as np

## Self-defined
from models.layer import *

class NNClassifier:
	def __init__(self, in_features, out_features, hidden_dim, activation, data_type):
		self.fc1 = FullyConnected(in_features, hidden_dim, activation, data_type)
		self.fc2 = FullyConnected(hidden_dim, hidden_dim, activation, data_type)
		self.fc3 = FullyConnected(hidden_dim, out_features, activation, data_type)
		self.act = Activation(activation)
		if activation.lower() == "none":
			self.out = Activation("sigmoid")
		else:
			self.out = Activation(activation)

	def forward(self, x):
		self.z1 = self.fc1.forward(x)
		self.a1 = self.act.forward(self.z1)
		self.z2 = self.fc2.forward(self.a1)
		self.a2 = self.act.forward(self.z2)
		self.z3 = self.fc3.forward(self.a2)
		self.y  = self.out.forward(self.z3)
		return self.y

	def backward(self, grad0):
		"""
		grad1 = self.out.backward(self.y ) * grad0
		grad2 = self.fc3.backward(grad1)
		grad3 = self.act.backward(self.a2) * grad2
		grad4 = self.fc2.backward(grad3)
		grad5 = self.act.backward(self.a1) * grad4
		grad6 = self.fc1.backward(grad5)
		"""
		grad1 = self.out.backward(grad0)
		grad2 = self.fc3.backward(grad1)
		grad3 = self.act.backward(grad2)
		grad4 = self.fc2.backward(grad3)
		grad5 = self.act.backward(grad4)
		grad6 = self.fc1.backward(grad5)

	def update(self, lr):
		self.fc1.update(lr)
		self.fc2.update(lr)
		self.fc3.update(lr)