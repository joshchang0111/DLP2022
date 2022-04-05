import ipdb
import torch
import torch.nn as nn

class EEGNet(nn.Module):
	def __init__(self, activation="relu", dropout_p=None):
		super(EEGNet, self).__init__()

		## Select activation function
		if activation == "relu":
			self.activation = nn.ReLU()
		elif activation == "leaky_relu":
			self.activation = nn.LeakyReLU()
		elif activation == "elu":
			self.activation = nn.ELU()

		## Set dropout probability
		self.dropout_p = 0.25 if dropout_p is None else dropout_p

		## EEGNet layers
		self.firstConv = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(1, 51), padding=(0, 25), bias=False), 
			nn.BatchNorm2d(16)
		)
		self.depthwiseConv = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=(2, 1), groups=16, bias=False), 
			nn.BatchNorm2d(32), 
			self.activation, 
			nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)), 
			nn.Dropout(p=self.dropout_p)
		)
		self.separableConv = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=(1, 15), padding=(0, 7), bias=False), 
			nn.BatchNorm2d(32), 
			self.activation, 
			nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)), 
			nn.Dropout(p=self.dropout_p)
		)
		self.classify = nn.Sequential(
			nn.Linear(736, 2)
		)

	def forward(self, x):
		## x.shape = (bs, 1, 2, 750)
		x = self.firstConv(x) ## (bs, 16, 2, 750)
		x = self.depthwiseConv(x) ## (bs, 32, 1, 187)
		x = self.separableConv(x) ## (bs, 32, 1, 23)
		
		## Flatten
		x = x.view(x.shape[0], -1) ## (bs, 736)
		y = self.classify(x) ## (bs, 2)
		return y
