import os
import ipdb
import random
import argparse
import itertools
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

## Self-defined
from data.loader import load_data
from models.build_models import build_models
from others.utils import kl_criterion, plot_pred, plot_rec, finn_eval_seq, pred

torch.backends.cudnn.benchmark = True

def parse_args():
	parser = argparse.ArgumentParser()

	## Hyper-parameters
	parser.add_argument("--cuda"      , default=False , action="store_true")
	parser.add_argument("--seed"      , default=1     , type=int,    help="manual seed")
	parser.add_argument("--lr"        , default=0.002 , type=float,  help="learning rate")
	parser.add_argument("--beta1"     , default=0.9   , type=float,  help="momentum term for adam")
	parser.add_argument("--batch_size", default=12    , type=int,    help="batch size")
	parser.add_argument("--optimizer" , default="adam",              help="optimizer to train with")
	parser.add_argument("--niter"     , type=int      , default=300, help="number of epochs to train for")
	parser.add_argument("--epoch_size", type=int      , default=600, help="epoch size")

	## Training strategies
	parser.add_argument("--tfr",                   type=float, default=1.0, help="teacher forcing ratio (0 ~ 1)")
	parser.add_argument("--tfr_start_decay_epoch", type=int  , default=0  , help="The epoch that teacher forcing ratio become decreasing")
	parser.add_argument("--tfr_decay_step"       , type=float, default=0  , help="The decay step size of teacher forcing ratio (0 ~ 1)")
	parser.add_argument("--tfr_lower_bound"      , type=float, default=0  , help="The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)")
	parser.add_argument("--kl_anneal_cyclical", default=False, action="store_true", help="use cyclical mode")
	parser.add_argument("--kl_anneal_ratio"   , type=float   , default=2          , help="The decay ratio of kl annealing")
	parser.add_argument("--kl_anneal_cycle"   , type=int     , default=3          , help="The number of cycle for kl annealing (if use cyclical mode)")
	parser.add_argument("--n_past"  , type=int, default=2  , help="number of frames to condition on")
	parser.add_argument("--n_future", type=int, default=10 , help="number of frames to predict")
	parser.add_argument("--n_eval"  , type=int, default=30 , help="number of frames to predict at eval time")
	parser.add_argument("--rnn_size", type=int, default=256, help="dimensionality of hidden layer")
	parser.add_argument("--posterior_rnn_layers", type=int, default=1, help="number of layers")
	parser.add_argument("--predictor_rnn_layers", type=int, default=2, help="number of layers")
	parser.add_argument("--z_dim", type=int  , default=64    , help="dimensionality of z_t")
	parser.add_argument("--g_dim", type=int  , default=128   , help="dimensionality of encoder output vector and decoder input vector")
	parser.add_argument("--beta" , type=float, default=0.0001, help="weighting on KL to prior")

	parser.add_argument("--num_workers"    , type=int, default=4, help="number of data loading threads")
	parser.add_argument("--last_frame_skip", action="store_true", help="if true, skip connections go between frame t and frame t+t rather than last ground truth frame")

	## Paths
	parser.add_argument("--log_dir"  , default="../logs/fp", help="base directory to save logs")
	parser.add_argument("--model_dir", default=""          , help="base directory to save logs")
	parser.add_argument("--data_root", default="../dataset", help="root directory for data")
	#parser.add_argument("--data_root", default='../data/processed_data', help='root directory for data')

	parser.add_argument("--debug", default=False, action="store_true")

	args = parser.parse_args()
	return args

def train(x, cond, modules, optimizer, kl_anneal, args):
	modules['frame_predictor'].zero_grad()
	modules['posterior'].zero_grad()
	modules['encoder'].zero_grad()
	modules['decoder'].zero_grad()

	## Initialize the hidden state.
	modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
	modules['posterior'].hidden = modules['posterior'].init_hidden()
	mse = 0
	kld = 0
	use_teacher_forcing = True if random.random() < args.tfr else False
	#ipdb.set_trace()
	for i in range(1, args.n_past + args.n_future):
		raise NotImplementedError

	beta = kl_anneal.get_beta()
	loss = mse + kld * beta
	loss.backward()

	optimizer.step()

	return loss.detach().cpu().numpy() / (args.n_past + args.n_future), mse.detach().cpu().numpy() / (args.n_past + args.n_future), kld.detach().cpu().numpy() / (args.n_future + args.n_past)

class kl_annealing():
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.beta = self.args.beta
		self.kl_anneal_cyclical = self.args.kl_anneal_cyclical
		self.kl_anneal_ratio = self.args.kl_anneal_ratio
		self.kl_anneal_cycle = self.args.kl_anneal_cycle
		#raise NotImplementedError
	
	def update(self):
		raise NotImplementedError
	
	def get_beta(self):
		return self.beta
		#raise NotImplementedError

def main():
	args = parse_args()

	if args.debug:
		args.num_workers = 0

	## Set device
	if args.cuda:
		assert torch.cuda.is_available(), "CUDA is not available."
		device = "cuda"
	else:
		device = "cpu"
	print("\nDevice: {}".format(device))
	
	assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
	assert 0 <= args.tfr and args.tfr <= 1
	assert 0 <= args.tfr_start_decay_epoch 
	assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

	if args.model_dir != "":
		## Load model and continue training from checkpoint
		saved_model = torch.load("%s/model.pth" % args.model_dir)
		optimizer = args.optimizer
		model_dir = args.model_dir
		niter = args.niter
		args = saved_model["args"]
		args.optimizer = optimizer
		args.model_dir = model_dir
		args.log_dir = "%s/continued" % args.log_dir
		start_epoch = saved_model["last_epoch"]
	else:
		saved_model = None
		name = "rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f" \
			% (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta)

		args.log_dir = "%s/%s" % (args.log_dir, name)
		niter = args.niter
		start_epoch = 0

	os.makedirs(args.log_dir, exist_ok=True)
	os.makedirs("%s/gen/" % args.log_dir, exist_ok=True)

	## Set seed
	print("Random Seed: {}".format(args.seed))
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	if os.path.exists("{}/train_record.txt".format(args.log_dir)):
		os.remove("{}/train_record.txt".format(args.log_dir))
	
	print("\nArguments:\n{}".format(args))

	with open("{}/train_record.txt".format(args.log_dir), "a") as train_record:
		train_record.write("args: {}\n".format(args))

	##################
	## Build Models ##
	##################
	frame_predictor, posterior, encoder, decoder = build_models(args, saved_model, device)

	##################
	## Load Dataset ##
	##################
	train_data, train_loader, train_iterator, \
	validate_data, validate_loader, validate_iterator = load_data(args)

	################
	## Optimizers ##
	################
	if args.optimizer == "adam":
		args.optimizer = optim.Adam
	elif args.optimizer == "rmsprop":
		args.optimizer = optim.RMSprop
	elif args.optimizer == "sgd":
		args.optimizer = optim.SGD
	else:
		raise ValueError("Unknown optimizer: %s" % args.optimizer)

	params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
	optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))
	kl_anneal = kl_annealing(args)

	modules = {
		"frame_predictor": frame_predictor,
		"posterior": posterior,
		"encoder": encoder,
		"decoder": decoder,
	}

	###################
	## Training Loop ##
	###################
	print("\nStart training...")
	progress = tqdm(total=args.niter)
	best_val_psnr = 0
	for epoch in range(start_epoch, start_epoch + niter):
		frame_predictor.train()
		posterior.train()
		encoder.train()
		decoder.train()

		epoch_loss = 0
		epoch_mse = 0
		epoch_kld = 0

		for _ in range(args.epoch_size):
			try:
				## Train on next batch
				seq, cond = next(train_iterator)
			except StopIteration:
				## If all batches have been trained, return to the first batch
				train_iterator = iter(train_loader)
				seq, cond = next(train_iterator)
			
			## Train a batch of sequences
			loss, mse, kld = train(seq, cond, modules, optimizer, kl_anneal, args)
			epoch_loss += loss
			epoch_mse += mse
			epoch_kld += kld
		
		if epoch >= args.tfr_start_decay_epoch:
			### Update teacher forcing ratio ###
			#raise NotImplementedError
			tfr_new = args.tfr - args.tfr_decay_step
			if tfr_new >= args.tfr_lower_bound:
				args.tfr = tfr_new

		progress.update(1)
		with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
			train_record.write(('[epoch: %02d] loss: %.5f | mse loss: %.5f | kld loss: %.5f\n' % (epoch, epoch_loss  / args.epoch_size, epoch_mse / args.epoch_size, epoch_kld / args.epoch_size)))
		
		################
		## Validation ##
		################
		frame_predictor.eval()
		encoder.eval()
		decoder.eval()
		posterior.eval()

		if epoch % 5 == 0:
			psnr_list = []
			for _ in range(len(validate_data) // args.batch_size):
				try:
					validate_seq, validate_cond = next(validate_iterator)
				except StopIteration:
					validate_iterator = iter(validate_loader)
					validate_seq, validate_cond = next(validate_iterator)

				pred_seq = pred(validate_seq, validate_cond, modules, args, device)
				_, _, psnr = finn_eval_seq(validate_seq[args.n_past:], pred_seq[args.n_past:])
				psnr_list.append(psnr)
				
			ave_psnr = np.mean(np.concatenate(psnr))


			with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
				train_record.write(('====================== validate psnr = {:.5f} ========================\n'.format(ave_psnr)))

			if ave_psnr > best_val_psnr:
				best_val_psnr = ave_psnr
				# save the model
				torch.save({
					'encoder': encoder,
					'decoder': decoder,
					'frame_predictor': frame_predictor,
					'posterior': posterior,
					'args': args,
					'last_epoch': epoch},
					'%s/model.pth' % args.log_dir)

		if epoch % 20 == 0:
			try:
				validate_seq, validate_cond = next(validate_iterator)
			except StopIteration:
				validate_iterator = iter(validate_loader)
				validate_seq, validate_cond = next(validate_iterator)

			plot_pred(validate_seq, validate_cond, modules, epoch, args)
			plot_rec(validate_seq, validate_cond, modules, epoch, args)

if __name__ == '__main__':
	main()
		
