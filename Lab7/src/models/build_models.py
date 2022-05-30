import ipdb

import torch
import torch.nn as nn

## Self-defined
from models import infogan, cgan, wgan, wgan_large

def weights_init(m):
	"""Customized weights initialization"""
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)

def build_models(args, device):
	print("\nBuilding models...")
	print("GAN type: {}".format(args.gan_type))

	if args.gan_type == "infogan":
		netG = infogan.Generator(args, device)
		netD = infogan.ClassifierD(args, device)
		netQ = infogan.ClassifierQ(args, device)
		shared = infogan.Discriminator(args, device)
		
		if args.train:
			netG.apply(weights_init)
			netD.apply(weights_init)
			netQ.apply(weights_init)
			shared.apply(weights_init)
		elif args.test:
			print("Loading model checkpoints...")
			netG.load_state_dict(torch.load("{}/{}/Generator_{}.pth".format(args.model_dir, args.exp_name, args.checkpoint_epoch)))
			netD.load_state_dict(torch.load("{}/{}/ClassifierD_{}.pth".format(args.model_dir, args.exp_name, args.checkpoint_epoch)))
			netQ.load_state_dict(torch.load("{}/{}/ClassifierQ_{}.pth".format(args.model_dir, args.exp_name, args.checkpoint_epoch)))
			shared.load_state_dict(torch.load("{}/{}/Discriminator_{}.pth".format(args.model_dir, args.exp_name, args.checkpoint_epoch)))
		
		return (netG, netD, netQ, shared)

	elif args.gan_type == "cgan":
		netG = cgan.Generator(args, device)
		netD = cgan.Discriminator(args, device)

		if args.train:
			netG.apply(weights_init)
			netD.apply(weights_init)
		elif args.test:
			print("Loading model checkpoints...")
			netG.load_state_dict(torch.load("{}/{}/Generator_{}.pth".format(args.model_dir, args.exp_name, args.checkpoint_epoch)))
			netD.load_state_dict(torch.load("{}/{}/Discriminator_{}.pth".format(args.model_dir, args.exp_name, args.checkpoint_epoch)))

		return (netG, netD)

	elif args.gan_type == "wgan":
		netG = wgan.Generator(args, device)
		netD = wgan.Discriminator(args, device)

		if args.train:
			netG.apply(weights_init)
			netD.apply(weights_init)
		elif args.test:
			print("Loading model checkpoints...")
			print("{}/{}/Generator_{}.pth".format(args.model_dir, args.exp_name, args.checkpoint_epoch))
			print("{}/{}/Discriminator_{}.pth".format(args.model_dir, args.exp_name, args.checkpoint_epoch))
			try:
				netG.load_state_dict(torch.load("{}/{}/Generator_{}.pth".format(args.model_dir, args.exp_name, args.checkpoint_epoch)))
				netD.load_state_dict(torch.load("{}/{}/Discriminator_{}.pth".format(args.model_dir, args.exp_name, args.checkpoint_epoch)))
			except:
				raise ValueError("Model checkpoints do not exist!")

		return (netG, netD)

	elif args.gan_type == "wgan-large":
		netG = wgan_large.Generator(args, device)
		netD = wgan_large.Discriminator(args, device)

		if args.train:
			netG.apply(weights_init)
			netD.apply(weights_init)
		elif args.test:
			print("Loading model checkpoints...")
			print("{}/{}/Generator_{}.pth".format(args.model_dir, args.exp_name, args.checkpoint_epoch))
			print("{}/{}/Discriminator_{}.pth".format(args.model_dir, args.exp_name, args.checkpoint_epoch))
			try:
				netG.load_state_dict(torch.load("{}/{}/Generator_{}.pth".format(args.model_dir, args.exp_name, args.checkpoint_epoch)))
				netD.load_state_dict(torch.load("{}/{}/Discriminator_{}.pth".format(args.model_dir, args.exp_name, args.checkpoint_epoch)))
			except:
				raise ValueError("Model checkpoints do not exist!")

		return (netG, netD)


