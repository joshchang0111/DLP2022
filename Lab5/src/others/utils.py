import os
import ipdb
import math
import imageio
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from operator import pos
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

import torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image, make_grid

def kl_criterion(mu, logvar, args):
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	KLD /= args.batch_size
	return KLD
	
def eval_seq(gt, pred):
	T = len(gt)
	bs = gt[0].shape[0]
	ssim = np.zeros((bs, T))
	psnr = np.zeros((bs, T))
	mse = np.zeros((bs, T))
	for i in range(bs):
		for t in range(T):
			origin = gt[t][i]
			predict = pred[t][i]
			for c in range(origin.shape[0]):
				ssim[i, t] += ssim_metric(origin[c], predict[c])
				psnr[i, t] += psnr_metric(origin[c], predict[c])
			ssim[i, t] /= origin.shape[0]
			psnr[i, t] /= origin.shape[0]
			mse[i, t] = mse_metric(origin, predict)

	return mse, ssim, psnr

def mse_metric(x1, x2):
	try:
		err = np.sum((x1 - x2) ** 2)
	except:
		err = torch.sum((x1 - x2) ** 2)
	err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
	return err

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
	T = len(gt)
	bs = gt[0].shape[0]
	ssim = np.zeros((bs, T))
	psnr = np.zeros((bs, T))
	mse = np.zeros((bs, T))
	for i in range(bs):
		for t in range(T):
			origin = gt[t][i].detach().cpu().numpy()
			predict = pred[t][i].detach().cpu().numpy()
			for c in range(origin.shape[0]):
				res = finn_ssim(origin[c], predict[c]).mean()
				if math.isnan(res):
					ssim[i, t] += -1
				else:
					ssim[i, t] += res
				psnr[i, t] += finn_psnr(origin[c], predict[c])
			ssim[i, t] /= origin.shape[0]
			psnr[i, t] /= origin.shape[0]
			mse[i, t] = mse_metric(origin, predict)

	return mse, ssim, psnr

def finn_psnr(x, y, data_range=1.):
	mse = ((x - y)**2).mean()
	return 20 * math.log10(data_range) - 10 * math.log10(mse)

def fspecial_gauss(size, sigma):
	x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
	g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
	return g / g.sum()

def finn_ssim(img1, img2, data_range=1., cs_map=False):
	img1 = img1.astype(np.float64)
	img2 = img2.astype(np.float64)

	size = 11
	sigma = 1.5
	window = fspecial_gauss(size, sigma)

	K1 = 0.01
	K2 = 0.03

	C1 = (K1 * data_range) ** 2
	C2 = (K2 * data_range) ** 2
	mu1 = signal.fftconvolve(img1, window, mode='valid')
	mu2 = signal.fftconvolve(img2, window, mode='valid')
	mu1_sq = mu1*mu1
	mu2_sq = mu2*mu2
	mu1_mu2 = mu1*mu2
	sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
	sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
	sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

	if cs_map:
		return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
					(sigma1_sq + sigma2_sq + C2)), 
				(2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
	else:
		return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
					(sigma1_sq + sigma2_sq + C2))

def init_weights(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1 or classname.find('Linear') != -1:
		m.weight.data.normal_(0.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

def plot_pred(validate_seq, validate_cond, modules, epoch, args, device):
	"""Plot predictions with z sampled from N(0, I)"""
	#raise NotImplementedError

	pred_seq = pred(validate_seq, validate_cond, modules, args, device)

	print("[Epoch {}] Saving predicted images & GIF...".format(epoch))
	os.makedirs("{}/gen/epoch-{}-pred".format(args.log_dir, epoch), exist_ok=True)

	## First one of this batch
	images, pred_frames, gt_frames = [], [], []
	sample_seq, gt_seq = pred_seq[:, 0, :, :, :], validate_seq[:, 0, :, :, :]
	for frame_idx in range(sample_seq.shape[0]):
		img_file = "{}/gen/epoch-{}-pred/{}.png".format(args.log_dir, epoch, frame_idx)
		save_image(sample_seq[frame_idx], img_file)
		images.append(imageio.imread(img_file))
		pred_frames.append(sample_seq[frame_idx])
		os.remove(img_file)

		gt_frames.append(gt_seq[frame_idx])

	pred_grid = make_grid(pred_frames, nrow=sample_seq.shape[0])
	gt_grid   = make_grid(gt_frames  , nrow=gt_seq.shape[0])
	save_image(pred_grid, "{}/gen/epoch-{}-pred/pred_grid.png".format(args.log_dir, epoch))
	save_image(gt_grid  , "{}/gen/epoch-{}-pred/gt_grid.png".format(args.log_dir, epoch))
	imageio.mimsave("{}/gen/epoch-{}-pred/animation.gif".format(args.log_dir, epoch), images)

def plot_rec(validate_seq, validate_cond, modules, epoch, args, device):
	"""Plot predictions with z sampled from encoder & gaussian_lstm"""
	#raise NotImplementedError

	## Transfer to device
	validate_seq  = validate_seq.to(device)
	validate_cond = validate_cond.to(device)

	with torch.no_grad():
		modules["frame_predictor"].hidden = modules["frame_predictor"].init_hidden()
		modules["posterior"].hidden = modules["posterior"].init_hidden()

		x_in = validate_seq[0]
		cond = validate_cond

		pred_seq = []
		pred_seq.append(x_in)

		## Iterate through 12 frames
		for frame_idx in range(1, args.n_past + args.n_future):
			## Encode the image at step (t-1)
			if args.last_frame_skip or frame_idx < args.n_past:
				h_in, skip = modules["encoder"](x_in)
			else:
				h_in, _    = modules["encoder"](x_in)

			## Obtain the latent vector z at step (t)
			h_t, _    = modules["encoder"](validate_seq[frame_idx])
			#z_t, _, _ = modules["posterior"](h_t)
			_, z_t, _ = modules["posterior"](h_t) ## Take the mean

			## Decode the image based on h_in & z_t
			if frame_idx < args.n_past:
				modules["frame_predictor"](torch.cat([h_in, z_t, cond[frame_idx - 1]], dim=1))
				x_in = validate_seq[frame_idx]
			else:
				g_t  = modules["frame_predictor"](torch.cat([h_in, z_t, cond[frame_idx - 1]], dim=1))
				x_in = modules["decoder"]([g_t, skip])

			pred_seq.append(x_in)

	pred_seq = torch.stack(pred_seq)

	print("[Epoch {}] Saving reconstructed images & GIF...".format(epoch))
	os.makedirs("{}/gen/epoch-{}-rec".format(args.log_dir, epoch), exist_ok=True)

	## First one of this batch
	images, frames = [], []
	sample_seq = pred_seq[:, 0, :, :, :]
	for frame_idx in range(sample_seq.shape[0]):
		img_file = "{}/gen/epoch-{}-rec/{}.png".format(args.log_dir, epoch, frame_idx)
		save_image(sample_seq[frame_idx], img_file)
		images.append(imageio.imread(img_file))
		frames.append(sample_seq[frame_idx])
		os.remove(img_file)

	grid = make_grid(frames, nrow=sample_seq.shape[0])
	save_image(grid, "{}/gen/epoch-{}-rec/rec_grid.png".format(args.log_dir, epoch))
	imageio.mimsave("{}/gen/epoch-{}-rec/animation.gif".format(args.log_dir, epoch), images)

def pred(validate_seq, validate_cond, modules, args, device):
	"""Predict on validation sequences"""
	#raise NotImplementedError

	## Transfer to device
	validate_seq  = validate_seq.to(device)
	validate_cond = validate_cond.to(device)

	with torch.no_grad():
		modules["frame_predictor"].hidden = modules["frame_predictor"].init_hidden()
		modules["posterior"].hidden = modules["posterior"].init_hidden()

		x_in = validate_seq[0]
		cond = validate_cond

		pred_seq = []
		pred_seq.append(x_in)

		## Iterate through 12 frames
		for frame_idx in range(1, args.n_past + args.n_future):
			## Encode the image at step (t-1)
			if args.last_frame_skip or frame_idx < args.n_past:
				h_in, skip = modules["encoder"](x_in)
			else:
				h_in, _    = modules["encoder"](x_in)

			## Obtain the latent vector z at step (t)
			if frame_idx < args.n_past:
				h_t, _    = modules["encoder"](validate_seq[frame_idx])
				#z_t, _, _ = modules["posterior"](h_t)
				_, z_t, _ = modules["posterior"](h_t) ## Take the mean
			else:
				z_t = torch.FloatTensor(args.batch_size, args.z_dim).normal_().to(device)

			## Decode the image based on h_in & z_t
			if frame_idx < args.n_past:
				modules["frame_predictor"](torch.cat([h_in, z_t, cond[frame_idx - 1]], dim=1))
				x_in = validate_seq[frame_idx]
			else:
				g_t  = modules["frame_predictor"](torch.cat([h_in, z_t, cond[frame_idx - 1]], dim=1))
				x_in = modules["decoder"]([g_t, skip])

			pred_seq.append(x_in)

	pred_seq = torch.stack(pred_seq)
	return pred_seq

def plot_params():
	"""Plot losses, psnr, kl annealing beta & teacher forcing ratio"""

	exp_name = "cyclical-bs20"

	records = {
		"epoch"     : [], 
		"loss"      : [], 
		"mse"       : [], 
		"kld"       : [], 
		"tfr"       : [], 
		"beta"      : [], 
		"epoch_psnr": [], 
		"psnr"      : []
	}

	with open("../logs/fp/{}/train_record.txt".format(exp_name)) as f_record:
		for line in f_record.readlines():
			line = line.strip().rstrip()
			if line.startswith("[epoch:"):
				epoch = int(line.split("]")[0].split(":")[-1].strip().rstrip()) + 1
				loss  = float(line.split("|")[0].split(":")[-1].strip().rstrip())
				mse   = float(line.split("|")[1].split(":")[-1].strip().rstrip())
				kld   = float(line.split("|")[2].split(":")[-1].strip().rstrip())
				tfr   = float(line.split("|")[3].split(":")[-1].strip().rstrip())
				beta  = float(line.split("|")[4].split(":")[-1].strip().rstrip())

				records["epoch"].append(epoch)
				records["loss"].append(loss)
				records["mse"].append(mse)
				records["kld"].append(kld)
				records["tfr"].append(tfr)
				records["beta"].append(beta)
			elif "validate psnr" in line:
				valid_psnr = float(line.replace("=", "").strip().rstrip().split(" ")[-1])

				records["epoch_psnr"].append(epoch)
				records["psnr"].append(valid_psnr)

	## Plot
	fig, main_ax = plt.subplots()
	sub_ax1 = main_ax.twinx()
	sub_ax2 = main_ax.twinx()

	cmap = plt.get_cmap("tab10")

	p1, = main_ax.plot(records["epoch"]     , records["loss"], color=cmap(0), label="Total Loss")
	p2, = main_ax.plot(records["epoch"]     , records["mse"] , color=cmap(1), label="MSE Loss")
	p3, = sub_ax1.plot(records["epoch"]     , records["kld"] , color=cmap(2), label="KLD Loss")
	p4, = sub_ax1.plot(records["epoch_psnr"], records["psnr"], color=cmap(3), label="PSNR")
	p5, = sub_ax2.plot(records["epoch"]     , records["tfr"] , color=cmap(4), linestyle=":", label="Teacher Forcing Ratio")
	p6, = sub_ax2.plot(records["epoch"]     , records["beta"], color=cmap(5), linestyle=":", label="KL Anneal Beta")

	main_ax.legend(handles=[p1, p2, p3, p4, p5, p6], loc="best")
	sub_ax2.spines["right"].set_position(("outward", 60))

	main_ax.set_xlabel("Epoch")
	main_ax.set_ylabel("Total / MSE Loss")
	sub_ax1.set_ylabel("KLD Loss / PSNR")
	sub_ax2.set_ylabel("Teacher Forcing Ratio / KL Anneal Beta")

	plt.tight_layout()
	plt.savefig("../logs/fp/{}/learning_curve.png".format(exp_name))

if __name__ == "__main__":
	plot_params()

