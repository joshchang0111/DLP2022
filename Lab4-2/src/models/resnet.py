import ipdb

import torch.nn as nn
from torchvision import models

class resnet(nn.Module):
	def __init__(self, model, pretrained, num_class):
		super(resnet, self).__init__()

		self.model_name = model
		self.num_class = num_class

		if self.model_name == "resnet18":
			self.resnet = models.resnet18(pretrained=pretrained)
		elif self.model_name == "resnet50":
			self.resnet = models.resnet50(pretrained=pretrained)

		## Reinitialize the last layer
		fc_in_dim  = self.resnet.fc.in_features
		fc_out_dim = self.num_class
		self.resnet.fc = nn.Linear(fc_in_dim, fc_out_dim)

	def forward(self, x):
		x = self.resnet(x)
		return x

	@property
	def device(self):
		return next(self.parameters()).device
