import ipdb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torchvision.utils import make_grid, save_image

## Self-defined
from others.evaluator import evaluation_model

def build_trainer(args, device, models):
	print("\nBuilding trainer...")
	trainer = Trainer(args, device, models)
	return trainer

class Trainer:
	def __init__(self, args, device, models):
		self.args = args
		self.device = device

		if self.args.gan_type == "infogan":
			self.netG = models[0]
			self.netD = models[1]
			self.netQ = models[2]
			self.shared = models[3]

			## Optimizer
			param_optimG = list(self.netG.parameters())
			param_optimD = list(self.shared.parameters()) + list(self.netD.parameters())
			param_optimQ = list(self.netQ.parameters())
			self.optimG = optim.Adam(param_optimG, lr=self.args.lr_G, betas=(self.args.beta1, self.args.beta2))
			self.optimD = optim.Adam(param_optimD, lr=self.args.lr_D, betas=(self.args.beta1, self.args.beta2))
			self.optimQ = optim.Adam(param_optimQ, lr=self.args.lr_D, betas=(self.args.beta1, self.args.beta2))

		elif self.args.gan_type == "cgan" or self.args.gan_type == "wgan" or self.args.gan_type == "wgan-large":
			self.netG = models[0]
			self.netD = models[1]

			## Optimizer
			self.optimG = optim.Adam(self.netG.parameters(), lr=self.args.lr_G, betas=(self.args.beta1, self.args.beta2))
			self.optimD = optim.Adam(self.netD.parameters(), lr=self.args.lr_D, betas=(self.args.beta1, self.args.beta2))
		
		## Create classification model for evaluation
		self.evaluator = evaluation_model(self.args)

		## Initialize log writer
		self.log_file = "{}/{}/log.txt".format(args.log_dir, args.exp_name)
		self.log_writer = open(self.log_file, "w")

	def train(self, train_loader, test_loader):
		"""Select different training procedures"""
		if self.args.gan_type == "infogan":
			self.train_infogan(train_loader, test_loader)
		elif self.args.gan_type == "cgan":
			self.train_cgan(train_loader, test_loader)
		elif self.args.gan_type == "wgan" or self.args.gan_type == "wgan-large":
			self.train_wgan(train_loader, test_loader)

	def compute_gradient_penalty(self, real, fake, cond):
		"""Calculates the gradient penalty loss for WGAN GP"""
		## Random weight term for interpolation between real and fake samples
		alpha = torch.rand(real.shape[0], 1, 1, 1).to(self.device)
		
		## Get random interpolation between real and fake samples
		interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
		d_interpolates, _ = self.netD(interpolates, cond)
		
		## Get gradient w.r.t. interpolates
		gradients = autograd.grad(
			inputs=interpolates,
			outputs=d_interpolates,
			grad_outputs=torch.ones(d_interpolates.shape, device=self.device),
			create_graph=True,
			retain_graph=True,
			only_inputs=True,
		)[0]
		gradients = gradients.view(gradients.size(0), -1)
		gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
		
		return gradient_penalty

	def train_wgan(self, train_loader, test_loader):
		"""Training loops for wgan"""

		G_losses, D_losses = [], []
		best_acc = 0

		test_cond = next(iter(test_loader)).to(self.device)
		#fixed_noise = torch.randn(test_cond.shape[0], self.args.z_dim, 1, 1, device=self.device)
		fixed_noise = [torch.randn(test_cond.shape[0], self.args.z_dim, 1, 1, device=self.device) for eval_ in range(self.args.n_eval)]
		fixed_noise = torch.stack(fixed_noise)
		torch.save(fixed_noise, "{}/{}/fixed_noise.pt".format(self.args.model_dir, self.args.exp_name, self.args.checkpoint_epoch))

		print("Start training {}...".format(self.args.gan_type))
		for epoch in range(self.args.epochs):

			for step, (img_real, cond) in enumerate(tqdm(train_loader, desc="[Epoch {:3d}]".format(epoch))):
				img_real = img_real.to(self.device)
				cond = cond.to(self.device)

				batch_len = img_real.shape[0]
				
				##########################
				## Update discriminator ##
				##########################
				loss_D_reported = 0
				for iter_critic in range(self.args.n_critic):
					self.netD.zero_grad()
					## Generate a batch of fake images
					noise = torch.randn(batch_len, self.args.z_dim, 1, 1, device=self.device)
					img_fake = self.netG(noise, cond)

					preds_real = self.netD(img_real, cond)
					preds_fake = self.netD(img_fake, cond)

					## Gradient Penalty!!! ##
					gp = self.compute_gradient_penalty(img_real, img_fake, cond)
					## Want to maximize EM-distance
					loss_D = -(torch.mean(preds_real) - torch.mean(preds_fake)) + self.args.lambda_gp * gp
					loss_D.backward(retain_graph=True)
					self.optimD.step()

					## Update statistics
					loss_D_reported = loss_D_reported + loss_D

				######################
				## Update generator ##
				######################
				self.netG.zero_grad()
				### Generate a batch of fake images
				img_fake = self.netG(noise, cond)
				preds_fake = self.netD(img_fake, cond)

				loss_G = -torch.mean(preds_fake)
				loss_G.backward()
				self.optimG.step()

				if step % self.args.report_freq == 0:
					#print("[Epoch {:3d}] Loss D: {:.4f}, Loss G: {:.4f}".format(epoch, loss_D.item(), loss_G.item()))
					print("[Epoch {:3d}]\tLoss D: {:.4f}\tLoss G: {:.4f}".format(epoch, loss_D_reported.item() / self.args.n_critic, loss_G.item()))
					self.log_writer.write("[Epoch {:3d}]\tLoss D: {:.4f}\tLoss G: {:.4f}\n".format(epoch, loss_D_reported.item() / self.args.n_critic, loss_G.item()))

					## Evaluate classification results
					eval_accs, best_eval_acc, best_pred_img = [], 0, None
					for eval_iter in range(self.args.n_eval):
						self.netG.eval()
						self.netD.eval()
						with torch.no_grad():
							pred_img = self.netG(fixed_noise[eval_iter], test_cond)
						eval_acc = self.evaluator.eval(pred_img, test_cond)
						eval_accs.append(eval_acc)

						if eval_acc > best_eval_acc:
							best_eval_acc = eval_acc
							best_pred_img = pred_img
					avg_acc = sum(eval_accs) / len(eval_accs)
					print("[Epoch {:3d}]\tAccuracy: {:.4f}".format(epoch, avg_acc))
					self.log_writer.write("[Epoch {:3d}]\tAccuracy: {:.4f}\n".format(epoch, avg_acc))
					
					## Save generated images
					save_image(pred_img, "{}/{}/pred_{}-{}.png".format(self.args.result_dir, self.args.exp_name, epoch, step), normalize=True)
					
					## Save model checkpoint
					if avg_acc > best_acc:
						best_acc = avg_acc
						print("[Epoch {:3d}]\tSaving model checkpoints with best accuracy...".format(epoch))
						torch.save(self.netG.state_dict(), "{}/{}/Generator_{}-{}.pth".format(self.args.model_dir, self.args.exp_name, epoch, step))
						torch.save(self.netD.state_dict(), "{}/{}/Discriminator_{}-{}.pth".format(self.args.model_dir, self.args.exp_name, epoch, step))

				G_losses.append(loss_G.item())
				D_losses.append(loss_D.item())

		self.log_writer.close()

	def train_cgan(self, train_loader, test_loader):
		"""Training loops for cgan"""
		G_losses, D_losses = [], []
		best_acc = 0

		criterion = nn.BCELoss()

		test_cond = next(iter(test_loader)).to(self.device)
		#fixed_noise = torch.randn(test_cond.shape[0], self.args.z_dim, 1, 1, device=self.device)
		fixed_noise = [torch.randn(test_cond.shape[0], self.args.z_dim, 1, 1, device=self.device) for eval_ in range(self.args.n_eval)]
		fixed_noise = torch.stack(fixed_noise)
		torch.save(fixed_noise, "{}/{}/fixed_noise.pt".format(self.args.model_dir, self.args.exp_name, self.args.checkpoint_epoch))

		print("Start training {}...".format(self.args.gan_type))
		for epoch in range(self.args.epochs):

			for step, (img, cond) in enumerate(tqdm(train_loader, desc="[Epoch {:3d}]".format(epoch))):
				img  = img.to(self.device)
				cond = cond.to(self.device)

				batch_len = img.shape[0]

				real_label = torch.ones( batch_len, device=self.device)
				fake_label = torch.zeros(batch_len, device=self.device)
				
				##########################
				## Update discriminator ##
				##########################
				## Train all-real batch
				self.netD.zero_grad()
				preds = self.netD(img, cond)
				loss_D_real = criterion(preds.flatten(), real_label)

				## Generate fake & train all-fake batch
				noise = torch.randn(batch_len, self.args.z_dim, 1, 1, device=self.device)
				fake  = self.netG(noise, cond)
				preds = self.netD(fake.detach(), cond)
				loss_D_fake = criterion(preds.flatten(), fake_label)
				loss_D = loss_D_real + loss_D_fake
				loss_D.backward()
				self.optimD.step()

				######################
				## Update generator ##
				######################
				for _ in range(4):
					self.netG.zero_grad()
					noise = torch.randn(batch_len, self.args.z_dim, 1, 1, device=self.device)
					fake  = self.netG(noise, cond)
					preds = self.netD(fake, cond)
					
					loss_G = criterion(preds.flatten(), real_label)
					loss_G.backward()
					self.optimG.step()

				if step % self.args.report_freq == 0:
					print("[Epoch {:3d}] Loss D: {:.4f}, Loss G: {:.4f}".format(epoch, loss_D.item(), loss_G.item()))

					### Evaluate classification results
					#self.netG.eval()
					#self.netD.eval()
					#with torch.no_grad():
					#	pred_img = self.netG(fixed_noise, test_cond)
					#acc = self.evaluator.eval(pred_img, test_cond)
					#print("[Epoch {:3d}] Accuracy: {:.4f}".format(epoch, acc))

					## Evaluate classification results
					eval_accs, best_eval_acc, best_pred_img = [], 0, None
					for eval_iter in range(self.args.n_eval):
						self.netG.eval()
						self.netD.eval()
						with torch.no_grad():
							pred_img = self.netG(fixed_noise[eval_iter], test_cond)
						eval_acc = self.evaluator.eval(pred_img, test_cond)
						eval_accs.append(eval_acc)

						if eval_acc > best_eval_acc:
							best_eval_acc = eval_acc
							best_pred_img = pred_img
					avg_acc = sum(eval_accs) / len(eval_accs)
					print("[Epoch {:3d}]\tAccuracy: {:.4f}".format(epoch, avg_acc))
					
					## Save generated images
					save_image(pred_img, "{}/{}/pred_{}-{}.png".format(self.args.result_dir, self.args.exp_name, epoch, step), normalize=True)
					
					## Save model checkpoint
					if avg_acc > best_acc:
						best_acc = avg_acc
						print("[Epoch {:3d}] Saving model checkpoints with best accuracy...".format(epoch))
						torch.save(self.netG.state_dict(), "{}/{}/Generator_{}-{}.pth".format(self.args.model_dir, self.args.exp_name, epoch, step))
						torch.save(self.netD.state_dict(), "{}/{}/Discriminator_{}-{}.pth".format(self.args.model_dir, self.args.exp_name, epoch, step))

				G_losses.append(loss_G.item())
				D_losses.append(loss_D.item())

	def train_infogan(self, train_loader, test_loader):
		"""Training loops for infogan"""
		G_losses, D_losses, Q_losses = [], [], []
		best_acc = 0

		criterion = nn.BCELoss()

		test_cond = next(iter(test_loader)).to(self.device)
		fixed_noise = torch.randn(test_cond.shape[0], self.args.z_dim, 1, 1, device=self.device)

		print("Start training {}...".format(self.args.gan_type))
		for epoch in range(self.args.epochs):

			for step, (img, cond) in enumerate(tqdm(train_loader, desc="[Epoch {:3d}]".format(epoch))):
				img  = img.to(self.device)
				cond = cond.to(self.device)

				batch_len = img.shape[0]

				real_label = torch.ones( batch_len, device=self.device)
				fake_label = torch.zeros(batch_len, device=self.device)
				
				##########################
				## Update discriminator ##
				##########################
				## Train all-real batch
				self.netD.zero_grad()
				self.shared.zero_grad()
				preds = self.netD(self.shared(img))
				loss_D_real = criterion(preds.flatten(), real_label)
				loss_D_real.backward()

				## Generate fake & train all-fake batch
				noise = torch.randn(batch_len, self.args.z_dim, 1, 1, device=self.device)
				fake  = self.netG(noise, cond)
				preds = self.netD(self.shared(fake.detach()))
				loss_D_fake = criterion(preds.flatten(), fake_label)
				loss_D_fake.backward()
				loss_D = loss_D_real + loss_D_fake

				## Update discriminator
				self.optimD.step()

				##########################
				## Update Generator & Q ##
				##########################
				self.netG.zero_grad()
				self.netQ.zero_grad()
				s_fake = self.shared(fake)
				preds  = self.netD(s_fake)
				q_fake = self.netQ(s_fake)

				s_real = self.shared(img)
				q_real = self.netQ(s_real)

				#loss_G_rec = criterion(preds.flatten(), real_label)
				#loss_G_aux = criterion(q_out, cond)
				#loss_G = loss_G_rec + self.args.lambda_Q * loss_G_aux
				#loss_G.backward()
				loss_G = criterion(preds.flatten(), real_label)
				
				loss_Q_fake = criterion(q_fake, cond)
				loss_Q_real = criterion(q_real, cond)
				loss_Q = loss_Q_fake + loss_Q_real
				
				loss_G_Q = loss_G + loss_Q
				loss_G_Q.backward()

				## Update generator & Q
				self.optimG.step()
				self.optimQ.step()

				if step % self.args.report_freq == 0:
					print("[Epoch {:3d}] Loss D: {:.4f}, Loss G: {:.4f}, Loss Q: {:.4f}".format(epoch, loss_D.item(), loss_G.item(), loss_Q.item()))
				G_losses.append(loss_G.item())
				D_losses.append(loss_D.item())
				Q_losses.append(loss_Q.item())

			## Evaluate classification results
			self.netG.eval()
			self.netD.eval()
			self.netQ.eval()
			self.shared.eval()
			with torch.no_grad():
				pred_img = self.netG(fixed_noise, test_cond)
			acc = self.evaluator.eval(pred_img, test_cond)
			print("[Epoch {:3d}] Accuracy: {:.4f}".format(epoch, acc))
			
			## Save generated images
			if (epoch % self.args.save_img_freq == 0) or (self.args.epochs - 1 == epoch):
				save_image(pred_img, "{}/{}/pred_{}.png".format(self.args.result_dir, self.args.exp_name, epoch), normalize=True)
			
			## Save model checkpoint
			if acc > best_acc:
				best_acc = acc
				print("[Epoch {:3d}] Saving model checkpoints with best accuracy...".format(epoch))
				torch.save(self.netG.state_dict(), "{}/{}/Generator_{}.pth".format(self.args.model_dir, self.args.exp_name, epoch))
				torch.save(self.netD.state_dict(), "{}/{}/ClassifierD_{}.pth".format(self.args.model_dir, self.args.exp_name, epoch))
				torch.save(self.netQ.state_dict(), "{}/{}/ClassifierQ_{}.pth".format(self.args.model_dir, self.args.exp_name, epoch))
				torch.save(self.shared.state_dict(), "{}/{}/Discriminator_{}.pth".format(self.args.model_dir, self.args.exp_name, epoch))

	def test(self, test_loader):
		"""Test only"""
		print("Start testing...")

		test_cond = next(iter(test_loader)).to(self.device)
		try:
			fixed_noise = torch.load("{}/{}/fixed_noise.pt".format(self.args.model_dir, self.args.exp_name, self.args.checkpoint_epoch)).to(self.device)
		except:
			print("`fixed_noise.pt` not found, try initializing random noise...")
			fixed_noise = torch.randn(test_cond.shape[0], self.args.z_dim, 1, 1, device=self.device)
		#fixed_noise = fixed_noise = [torch.randn(test_cond.shape[0], self.args.z_dim, 1, 1, device=self.device) for eval_ in range(100)]
		#fixed_noise = torch.stack(fixed_noise)

		if len(fixed_noise.shape) == 4:
			fixed_noise = torch.stack([fixed_noise])

		best_acc, best_pred_img = 0, None
		for eval_iter in range(len(fixed_noise)):
			## Evaluate classification results
			self.netG.eval()
			self.netD.eval()
			with torch.no_grad():
				pred_img = self.netG(fixed_noise[eval_iter], test_cond)
			acc = self.evaluator.eval(pred_img, test_cond)
			print("Accuracy: {:.4f}".format(acc))

			if acc > best_acc:
				best_acc = acc
				best_pred_img = pred_img

		save_image(best_pred_img, "{}/{}/pred_{:.4f}.png".format(self.args.result_dir, self.args.exp_name, best_acc), normalize=True)



