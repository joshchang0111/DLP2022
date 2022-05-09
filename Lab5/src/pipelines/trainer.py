import ipdb
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

## Self-defined
from others.utils import mse_metric, kl_criterion, plot_pred, plot_rec, finn_eval_seq, pred

mse_criterion = nn.MSELoss()

def build_trainer(args, frame_predictor, posterior, encoder, decoder, device):
	print("\nBuilding trainer...")

	trainer = Trainer(
		args, 
		frame_predictor, 
		posterior, 
		encoder, 
		decoder, 
		device
	)
	return trainer

class kl_annealing():
	def __init__(self, args):
		super().__init__()
		#raise NotImplementedError

		self.args = args
		self.beta = self.args.beta
		self.kl_anneal_cyclical = self.args.kl_anneal_cyclical
		self.kl_anneal_ratio = self.args.kl_anneal_ratio
		self.kl_anneal_cycle = self.args.kl_anneal_cycle
		if self.kl_anneal_cyclical:
			self.period = int(self.args.niter / self.kl_anneal_cycle)
		else:
			self.period = self.args.niter
	
	def update(self, epoch):
		#raise NotImplementedError

		if (epoch % self.period) <= (self.period / self.kl_anneal_ratio):
			## Reset if cycle reached when in cyclical mode
			#if self.kl_anneal_cyclical:
			if epoch % self.period == 0:
				self.beta = 0
			else:
				self.step = (1 - 0) / (self.period / self.kl_anneal_ratio)
				self.beta = self.beta + self.step
		else:
			self.beta = 1
	
	def get_beta(self):
		#raise NotImplementedError
		return self.beta

class Trainer:
	def __init__(self, args, frame_predictor, posterior, encoder, decoder, device):
		self.args = args
		self.device = device

		## Set optimizer
		if self.args.optimizer == "adam":
			self.args.optimizer = optim.Adam
		elif self.args.optimizer == "rmsprop":
			self.args.optimizer = optim.RMSprop
		elif self.args.optimizer == "sgd":
			self.args.optimizer = optim.SGD
		else:
			raise ValueError("Unknown optimizer: %s" % self.args.optimizer)

		params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
		self.optimizer = self.args.optimizer(params, lr=self.args.lr, betas=(self.args.beta1, 0.999))
		self.kl_anneal = kl_annealing(self.args)

		self.modules = {
			"frame_predictor": frame_predictor,
			"posterior": posterior,
			"encoder": encoder,
			"decoder": decoder,
		}

	def train(
			self, start_epoch, niter, 
			train_data, train_loader, train_iterator, 
			valid_data, valid_loader, valid_iterator
		):
		"""Main training loop"""
		print("Start training...")
		progress = tqdm(total=self.args.niter)
		best_val_psnr = 0
		kl_betas, tfrs = [], []

		for epoch in range(start_epoch, start_epoch + niter):
			self.modules["frame_predictor"].train()
			self.modules["posterior"].train()
			self.modules["encoder"].train()
			self.modules["decoder"].train()
	
			epoch_loss = 0
			epoch_mse = 0
			epoch_kld = 0

			## Update KL annealing weight
			self.kl_anneal.update(epoch)

			for _ in tqdm(range(self.args.epoch_size), desc="[Epoch {}]".format(epoch)):
				try:
					## Train on next batch
					seq, cond = next(train_iterator)
				except StopIteration:
					## If all batches have been trained, return to the first batch
					train_iterator = iter(train_loader)
					seq, cond = next(train_iterator)

				## Swap axes of batch & frames
				seq  = seq.permute((1, 0, 2, 3, 4))
				cond = cond.permute((1, 0, 2))
				
				## Train a batch of sequences
				loss, mse, kld = self.train_batch(seq, cond)
				epoch_loss += loss
				epoch_mse += mse
				epoch_kld += kld
	
			progress.update(1)
			with open("{}/train_record.txt".format(self.args.log_dir), "a") as train_record:
				train_record.write(
					("[epoch: %02d] loss: %.5f | mse loss: %.5f | kld loss: %.5f | tf ratio: %.5f | kld beta: %.5f\n" % \
						(
							epoch, 
							epoch_loss / self.args.epoch_size, 
							epoch_mse  / self.args.epoch_size, 
							epoch_kld  / self.args.epoch_size, 
							self.args.tfr, 
							self.kl_anneal.get_beta()
						)
					)
				)

			### Update teacher forcing ratio ###
			if epoch >= self.args.tfr_start_decay_epoch:
				#raise NotImplementedError

				self.args.tfr_decay_step = (1 - 0) / self.args.niter
				self.args.tfr = self.args.tfr - self.args.tfr_decay_step
				if self.args.tfr < self.args.tfr_lower_bound:
					self.args.tfr = self.args.tfr + self.args.tfr_decay_step

			## Record kl annealing weight & teacher forcing ratio
			kl_betas.append(self.kl_anneal.get_beta())
			tfrs.append(self.args.tfr)
			#continue
			
			################
			## Validation ##
			################
			self.modules["frame_predictor"].eval()
			self.modules["posterior"].eval()
			self.modules["encoder"].eval()
			self.modules["decoder"].eval()
	
			if epoch % 5 == 0:
				print("\nRunning validation...")
				psnr_list = []
				for _ in tqdm(range(len(valid_data) // self.args.batch_size)):
					try:
						validate_seq, validate_cond = next(valid_iterator)
					except StopIteration:
						valid_iterator = iter(valid_loader)
						validate_seq, validate_cond = next(valid_iterator)

					validate_seq  = validate_seq.permute((1, 0, 2, 3, 4))[:self.args.n_past + self.args.n_future]
					validate_cond = validate_cond.permute((1, 0, 2))[:self.args.n_past + self.args.n_future]
	
					pred_seq = pred(validate_seq, validate_cond, self.modules, self.args, self.device)
					_, _, psnr = finn_eval_seq(validate_seq[self.args.n_past:], pred_seq[self.args.n_past:])
					psnr_list.append(psnr)
				
				ave_psnr = np.mean(np.concatenate(psnr_list))
	
				with open("{}/train_record.txt".format(self.args.log_dir), "a") as train_record:
					train_record.write(("====================== validate psnr = {:.5f} ========================\n".format(ave_psnr)))
	
				if ave_psnr > best_val_psnr:
					print("[Epoch {}] Saving model with best validation psnr...".format(epoch))
					best_val_psnr = ave_psnr
					## save the model
					torch.save(
						{
							"encoder": self.modules["encoder"],
							"decoder": self.modules["decoder"],
							"frame_predictor": self.modules["frame_predictor"],
							"posterior": self.modules["posterior"],
							"args": self.args,
							"last_epoch": epoch, 
							"best_val_psnr": best_val_psnr
						},
						"%s/model.pth" % self.args.log_dir
					)
					
			if epoch % 20 == 0:
				try:
					validate_seq, validate_cond = next(valid_iterator)
				except StopIteration:
					valid_iterator = iter(valid_loader)
					validate_seq, validate_cond = next(valid_iterator)

				validate_seq  = validate_seq.permute((1, 0, 2, 3, 4))[:self.args.n_past + self.args.n_future]
				validate_cond = validate_cond.permute((1, 0, 2))[:self.args.n_past + self.args.n_future]
				
				plot_pred(validate_seq, validate_cond, self.modules, epoch, self.args, self.device)
				plot_rec( validate_seq, validate_cond, self.modules, epoch, self.args, self.device)

	def train_batch(self, x, cond):
		"""Loop on a batch of sequences"""
		self.modules['frame_predictor'].zero_grad()
		self.modules['posterior'].zero_grad()
		self.modules['encoder'].zero_grad()
		self.modules['decoder'].zero_grad()
		
		## Initialize the hidden state.
		self.modules['frame_predictor'].hidden = self.modules['frame_predictor'].init_hidden()
		self.modules['posterior'].hidden = self.modules['posterior'].init_hidden()
		mse = 0
		kld = 0
		use_teacher_forcing = True if random.random() < self.args.tfr else False

		## Transfer input to device
		x    =    x.to(self.device)
		cond = cond.to(self.device)
		
		## Encoder all frames first
		h_seq = [self.modules["encoder"](x[i]) for i in range(self.args.n_past + self.args.n_future)]

		## Iterate through first 12 frames
		for i in range(1, self.args.n_past + self.args.n_future):
			## Get encoded information
			h_t, _ = h_seq[i]
			
			if self.args.last_frame_skip or (i < self.args.n_past):
				h_in, skip = h_seq[i - 1]
			else:
				if use_teacher_forcing:
					h_in, _    = h_seq[i - 1]
				else:
					h_in, _    = self.modules["encoder"](x_pred)
			
			## Obtain latent vector z
			z_t, mu, logvar = self.modules["posterior"](h_t)

			## Decode the image
			lstm_in = torch.cat([h_in, z_t, cond[i - 1]], dim=1)
			g_t     = self.modules["frame_predictor"](lstm_in)
			x_pred  = self.modules["decoder"]([g_t, skip])

			## Calculate loss values
			mse = mse + mse_criterion(x[i], x_pred)
			kld = kld + kl_criterion(mu, logvar, self.args)

		beta = self.kl_anneal.get_beta()
		loss = mse + kld * beta
		loss.backward()
		
		self.optimizer.step()
		
		return loss.detach().cpu().numpy() / (self.args.n_past + self.args.n_future), \
			   mse.detach().cpu().numpy()  / (self.args.n_past + self.args.n_future), \
			   kld.detach().cpu().numpy()  / (self.args.n_past + self.args.n_future)

	def test(self, test_data, test_loader, test_iterator, test_set="test"):
		"""Test only"""
		print("Testing only, plotting results...")
		psnr_list = []
		for _ in tqdm(range(len(test_data) // self.args.batch_size + 1)):
			try:
				test_seq, test_cond = next(test_iterator)
			except StopIteration:
				test_iterator = iter(test_loader)
				test_seq, test_cond = next(test_iterator)

			test_seq  = test_seq.permute((1, 0, 2, 3, 4))[:self.args.n_past + self.args.n_future]
			test_cond = test_cond.permute((1, 0, 2))[:self.args.n_past + self.args.n_future]

			pred_seq = pred(test_seq, test_cond, self.modules, self.args, self.device)
			_, _, psnr = finn_eval_seq(test_seq[self.args.n_past:], pred_seq[self.args.n_past:])
			psnr_list.append(psnr)

		ave_psnr = np.mean(np.concatenate(psnr_list))
		print("[Epoch best] {} psnr = {:.5f}".format(test_set, ave_psnr))

		sample_idx = np.random.randint(0, self.args.batch_size)

		plot_pred(test_seq, test_cond, self.modules, "best", self.args, self.device, sample_idx=sample_idx)
		plot_rec( test_seq, test_cond, self.modules, "best", self.args, self.device, sample_idx=sample_idx)

