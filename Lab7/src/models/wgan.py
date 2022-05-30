import ipdb

import torch
import torch.nn as nn

class Generator(nn.Module):
	"""Generator architecture based on DCGAN"""
	def __init__(self, args, device):
		super(Generator, self).__init__()
		self.args = args
		self.device = device

		## Layers
		self.cond_layer = nn.Sequential(
			nn.Linear(24, self.args.input_dim * self.args.input_dim), 
			nn.ReLU()
		)
		self.main = nn.Sequential(
			## Input: latent vector z -> convolution
			nn.ConvTranspose2d(self.args.z_dim + self.args.input_dim * self.args.input_dim, self.args.input_dim * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(self.args.input_dim * 8),
			nn.ReLU(True),

			## State size: (self.args.input_dim * 8) x 4 x 4
			nn.ConvTranspose2d(self.args.input_dim * 8, self.args.input_dim * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.args.input_dim * 4),
			nn.ReLU(True),
			
			## State size: (self.args.input_dim * 4) x 8 x 8
			nn.ConvTranspose2d(self.args.input_dim * 4, self.args.input_dim * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.args.input_dim * 2),
			nn.ReLU(True),
			
			## State size: (self.args.input_dim * 2) x 16 x 16
			nn.ConvTranspose2d(self.args.input_dim * 2, self.args.input_dim, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.args.input_dim),
			nn.ReLU(True),
			
			## State size: (self.args.input_dim) x 32 x 32
			nn.ConvTranspose2d(self.args.input_dim, self.args.n_channel, 4, 2, 1, bias=False),
			nn.Tanh()
			## State size: (self.args.n_channel) x 64 x 64
		)

		self.to(device)

	def forward(self, input, cond):
		cond_emb = self.cond_layer(cond)
		cond_emb = cond_emb.view(cond_emb.shape[0], cond_emb.shape[1], 1, 1)
		input = torch.cat([input, cond_emb], dim=1)
		return self.main(input)

class Discriminator(nn.Module):
	"""
	Conditional discriminator based on DCGAN
	"""
	def __init__(self, args, device):
		super(Discriminator, self).__init__()
		self.args = args
		self.device = device

		## Layers
		self.cond_layer = nn.Sequential(
			nn.Linear(24, self.args.input_dim * self.args.input_dim), 
			nn.LeakyReLU()
		)
		self.main = nn.Sequential(
			## Input size: (self.args.n_channel + 1) x 64 x 64
			nn.Conv2d(self.args.n_channel + 1, self.args.input_dim, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			## State size: (self.args.input_dim) x 32 x 32
			nn.Conv2d(self.args.input_dim, self.args.input_dim * 2, 4, 2, 1, bias=False),
			nn.InstanceNorm2d(self.args.input_dim * 2, affine=True),
			nn.LeakyReLU(0.2, inplace=True),
			
			## State size: (self.args.input_dim * 2) x 16 x 16
			nn.Conv2d(self.args.input_dim * 2, self.args.input_dim * 4, 4, 2, 1, bias=False),
			nn.InstanceNorm2d(self.args.input_dim * 4, affine=True),
			nn.LeakyReLU(0.2, inplace=True),
			
			## State size: (self.args.input_dim * 4) x 8 x 8
			nn.Conv2d(self.args.input_dim * 4, self.args.input_dim * 8, 4, 2, 1, bias=False),
			nn.InstanceNorm2d(self.args.input_dim * 8, affine=True),
			nn.LeakyReLU(0.2, inplace=True), 

			## State size: (self.args.input_dim * 8) x 4 x 4
			nn.Conv2d(self.args.input_dim * 8, 1, 4, 1, 0, bias=False), 
			#nn.Sigmoid()
		)

		self.to(device)

	def forward(self, input, cond):
		cond_emb = self.cond_layer(cond)
		cond_emb = cond_emb.view(cond_emb.shape[0], 1, self.args.input_dim, self.args.input_dim)
		input = torch.cat([input, cond_emb], dim=1)
		return self.main(input)




