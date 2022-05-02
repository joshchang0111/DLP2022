import os
import csv
import ipdb
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

default_transform = transforms.Compose([
	transforms.ToTensor(),
])

class dataset(Dataset):
	def __init__(self, args, mode="", transform=default_transform):
		assert mode == "train" or mode == "test" or mode == "validate"
		#raise NotImplementedError
		
	def set_seed(self, seed):
		if not self.seed_is_set:
			self.seed_is_set = True
			np.random.seed(seed)
			
	def __len__(self):
		raise NotImplementedError
		
	def get_seq(self):
		raise NotImplementedError
	
	def get_csv(self):
		raise NotImplementedError
	
	def __getitem__(self, index):
		self.set_seed(index)
		seq  = self.get_seq()
		cond = self.get_csv()
		return seq, cond

class bair_robot_pushing_dataset(dataset):
	"""Customized dataset"""
	def __init__(self, args, mode="train", transform=default_transform):
		super(bair_robot_pushing_dataset, self).__init__(args, mode, transform)

		self.args = args
		self.mode = mode
		self.transforms = transform

		## Get all data paths
		self.data_dirs = []
		self.mode_data_root = "{}/{}".format(self.args.data_root, self.mode)
		for data_dir in os.listdir(self.mode_data_root):
			if data_dir[0] == ".":
				continue
			idx_dirs = os.listdir("{}/{}".format(self.mode_data_root, data_dir))
			for idx_dir in idx_dirs:
				if idx_dir[0] == ".":
					continue
				self.data_dirs.append("{}/{}/{}".format(self.mode_data_root, data_dir, idx_dir))

		## Whether the random seed is already set or not
		self.seed_is_set = True

	def __len__(self):
		return len(self.data_dirs)

	def get_seq(self, index):
		"""Get the sequence of frames"""
		seqs = []
		frame_files = [file for file in os.listdir(self.data_dirs[index]) if ".png" in file and file[0] != "."]
		frame_files.sort(key=lambda x: int(x.split(".")[0]))
		for frame_file in frame_files:
			img_arr = Image.open("{}/{}".format(self.data_dirs[index], frame_file))
			img_tensor = self.transforms(img_arr)
			seqs.append(img_tensor)

		## Transform to tensor of shape (30, 3, 64, 64)
		seqs = torch.stack(seqs)
		return seqs

	def get_csv(self, index):
		"""Get the actions and positions as conditions"""
		conds = []
		files = [
			"{}/actions.csv".format(self.data_dirs[index]), 
			"{}/endeffector_positions.csv".format(self.data_dirs[index])
		]
		for csv_file in files:
			with open(csv_file) as f_csv:
				rows = csv.reader(f_csv)
				cond_arr = np.array(list(rows)).astype(float)
				cond_tensor = self.transforms(cond_arr)
			conds.append(cond_tensor)
		
		## Concatenate 2 conditions, transform to tensor of shape (30, 7)
		conds = torch.cat(conds, dim=2)[0]
		conds = conds.float()
		return conds

	def __getitem__(self, index):
		self.set_seed(index)
		seq  = self.get_seq(index)
		cond = self.get_csv(index)
		return seq, cond
