import os
import csv
import ipdb
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def parse_args():
	parser = argparse.ArgumentParser(description="Preprocess dataset.")

	## Mode
	parser.add_argument("--preprocess", action="store_true")
	parser.add_argument("--split_5_fold", action="store_true")
	parser.add_argument("--split_target", action="store_true")

	parser.add_argument("--n_fold", type=int, default=5)
	parser.add_argument("--n_target_fold", type=int, default=5)

	parser.add_argument("--dataset", type=str, default="IBM")
	parser.add_argument("--raw_path", type=str, default="../dataset/raw")
	parser.add_argument("--processed_path", type=str, default="../dataset/processed")

	args = parser.parse_args()

	return args

def preprocess_IBM(args):
	raw_file = "{}/{}/claim_stance_dataset_v1.csv".format(args.raw_path, args.dataset)
	processed_file = "{}/{}/data.csv".format(args.processed_path, args.dataset)

	write_rows = []
	with open(raw_file, "r") as csv_f:
		rows = csv.reader(csv_f)
		rows = list(rows)

		for row in tqdm(rows[1:]):
			row_dict = {}
			row_dict["split"] = row[1]
			row_dict["topic text"] = row[2]
			row_dict["claim text"] = row[7]
			row_dict["claim stance"] = row[6]
			row_dict["claim sentiment"] = row[19]
			write_rows.append(row_dict)

	with open(processed_file, "w") as csv_fw:
		writer = csv.writer(csv_fw)
		#writer.writerow(["split", "target", "claim", "stance", "sentiment"])
		writer.writerow(["target", "claim", "stance", "sentiment"])

		for row_dict in tqdm(write_rows):
			#write_row = [row_dict["split"], row_dict["topic text"], row_dict["claim text"], row_dict["claim stance"], row_dict["claim sentiment"]]
			write_row = [row_dict["topic text"], row_dict["claim text"], row_dict["claim stance"], row_dict["claim sentiment"]]
			writer.writerow(write_row)

def split_5_fold(args):
	"""Split processed data into 5-fold"""
	processed_file = "{}/{}/data.csv".format(args.processed_path, args.dataset)

	## Read in `data.csv`
	data_df = pd.read_csv(processed_file)

	folds = []
	stance_groups = data_df.groupby("stance")

	## Get 5 portions
	for fold_idx in range(args.n_fold):
		fold_of_stances = []
		for stance, group in stance_groups:
			## shuffle
			group = group.sample(frac=1).reset_index(drop=True)
			
			n_portion = int(len(group) / args.n_fold)
			start_idx = fold_idx * n_portion
			end_idx   = len(group) if fold_idx == args.n_fold - 1 else (fold_idx + 1) * n_portion
			fold_of_stances.append(group.iloc[start_idx:end_idx])

			#print("Fold {}, stance {}, start_idx: {:4d}, end_idx: {:4d}".format(fold_idx, stance, start_idx, end_idx))
		
		folds.append(pd.concat(fold_of_stances).fillna(0))

	## Write `train.csv` and `test.csv` for each split
	for fold_idx in range(args.n_fold):
		## Take turns be test set
		test = folds[fold_idx]

		## Get train set
		train = []
		for i in range(args.n_fold):
			if i != fold_idx:
				train.append(folds[i])
		train = pd.concat(train)

		## Write file
		split_path = "{}/{}/split_{}".format(args.processed_path, args.dataset, fold_idx)
		os.makedirs(split_path, exist_ok=True)
		train.to_csv("{}/train.csv".format(split_path), index=False)
		test.to_csv("{}/test.csv".format(split_path), index=False)

		## Get statistics
		print("\n***** Fold {} *****".format(fold_idx))
		train_target, test_target = train.groupby("target").size(), test.groupby("target").size()
		train_stance, test_stance = train.groupby("stance").size(), test.groupby("stance").size()
		train_sentim, test_sentim = train.groupby("sentiment").size(), test.groupby("sentiment").size()

		print("# train target: {:4d}, # test target: {:4d}".format(len(train_target), len(test_target)))
		print("# train    CON: {:4d}, # test    CON: {:4d}".format(train_stance["CON"], test_stance["CON"]))
		print("# train    PRO: {:4d}, # test    PRO: {:4d}".format(train_stance["PRO"], test_stance["PRO"]))
		#ipdb.set_trace()

def split_target(args):
	"""Split processed data for target-wise evaluation"""
	processed_file = "{}/{}/data.csv".format(args.processed_path, args.dataset)

	## Read in `data.csv`
	data_df = pd.read_csv(processed_file)

	targets = list(set(data_df["target"].tolist()))
	np.random.shuffle(targets)

	## Split 55 targets into 5 groups
	target_fold_len = int(len(targets) / args.n_target_fold)

	target_dict = {}
	for fold_idx in range(args.n_target_fold):
		target_dict[fold_idx] = targets[fold_idx * target_fold_len:(fold_idx + 1) * target_fold_len]

	## Get data for each target split
	folds = []
	target_groups = data_df.groupby("target")
	for fold_idx in target_dict.keys():
		fold_of_targets = []
		for target, group in target_groups:
			if target in target_dict[fold_idx]:
				fold_of_targets.append(group)
		folds.append(pd.concat(fold_of_targets).fillna(0))

	## Write `train.csv` and `test.csv` for each split
	for fold_idx in range(args.n_fold):
		## Take turns be test set
		test = folds[fold_idx]

		## Get train set
		train = []
		for i in range(args.n_fold):
			if i != fold_idx:
				train.append(folds[i])
		train = pd.concat(train)

		## Shuffle
		train = train.sample(frac=1).reset_index(drop=True)
		test  = test.sample(frac=1).reset_index(drop=True)

		## Write file
		split_path = "{}/{}/target_{}".format(args.processed_path, args.dataset, fold_idx)
		os.makedirs(split_path, exist_ok=True)
		train.to_csv("{}/train.csv".format(split_path), index=False)
		test.to_csv("{}/test.csv".format(split_path), index=False)

		## Get statistics
		print("\n***** Target {} *****".format(fold_idx))
		train_target, test_target = train.groupby("target").size(), test.groupby("target").size()
		train_stance, test_stance = train.groupby("stance").size(), test.groupby("stance").size()
		train_sentim, test_sentim = train.groupby("sentiment").size(), test.groupby("sentiment").size()

		print("# train target: {:4d}, # test target: {:4d}".format(len(train_target), len(test_target)))
		print("# train    CON: {:4d}, # test    CON: {:4d}".format(train_stance["CON"], test_stance["CON"]))
		print("# train    PRO: {:4d}, # test    PRO: {:4d}".format(train_stance["PRO"], test_stance["PRO"]))

def main(args):
	if args.preprocess:
		if args.dataset == "IBM":
			preprocess_IBM(args)
	elif args.split_5_fold:
		split_5_fold(args)
	elif args.split_target:
		split_target(args)

if __name__ == "__main__":
	np.random.seed(123)

	args = parse_args()
	main(args)