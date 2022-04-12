import ipdb
import torch
import torch.nn as nn

class DeepConvNet(nn.Module):
	def __init__(self, activation="relu", dropout_p=None):
		super(DeepConvNet, self).__init__()

		## Select activation function
		if activation == "relu":
			self.activation = nn.ReLU()
		elif activation == "leaky_relu":
			self.activation = nn.LeakyReLU()
		elif activation == "elu":
			self.activation = nn.ELU()

		## Set dropout probability
		self.dropout_p = 0.5 if dropout_p is None else dropout_p

		## Parameters for convs
		out_channels = [25, 25, 50, 100, 200]
		kernel_sizes = [(2, 1), (1, 5), (1, 5), (1, 5)]

		## DeepConvNet layers
		self.conv0 = nn.Conv2d(1, 25, kernel_size=(1, 5))
		self.convs = nn.ModuleList()
		for idx in range(4):
			conv_i = nn.Sequential(
				nn.Conv2d(out_channels[idx], out_channels[idx + 1], kernel_size=kernel_sizes[idx]), 
				nn.BatchNorm2d(out_channels[idx + 1]), 
				self.activation, 
				nn.MaxPool2d(kernel_size=(1, 2)), 
				nn.Dropout(p=self.dropout_p)
			)
			self.convs.append(conv_i)
		self.classify = nn.Linear(8600, 2)

	def forward(self, x):
		## x.shape = (bs, 1, 2, 750)
		x = self.conv0(x) ## (bs, 25, 2, 746)
		for conv_i in self.convs: ## (bs, 25, 1, 373) ## (bs, 50, 1, 184) ## (bs, 100, 1, 90) ## (bs, 200, 1, 43)
			x = conv_i(x)

		## Flatten
		x = x.view(x.shape[0], -1) ## (bs, 8600)
		y = self.classify(x) ## (bs, 2)
		return y
