import ipdb
from torch.utils.data import DataLoader

## Self-defined
from data.dataset import bair_robot_pushing_dataset

def load_train_data(args):
	print("\nLoading datasets...")

	train_data = bair_robot_pushing_dataset(args, "train")
	valid_data = bair_robot_pushing_dataset(args, "validate")

	train_loader = DataLoader(
		train_data,
		num_workers=args.num_workers,
		batch_size=args.batch_size,
		shuffle=True,
		drop_last=True,
		pin_memory=True
	)
	valid_loader = DataLoader(
		valid_data,
		num_workers=args.num_workers,
		batch_size=args.batch_size,
		shuffle=True,
		drop_last=True,
		pin_memory=True
	)

	train_iterator = iter(train_loader)
	valid_iterator = iter(valid_loader)

	return train_data, train_loader, train_iterator, \
		   valid_data, valid_loader, valid_iterator

def load_test_data(args):
	print("\nLoading test dataset...")

	test_data = bair_robot_pushing_dataset(args, "test")

	test_loader = DataLoader(
		test_data,
		num_workers=args.num_workers,
		batch_size=args.batch_size,
		shuffle=True,
		drop_last=True,
		pin_memory=True
	)

	test_iterator = iter(test_loader)

	return test_data, test_loader, test_iterator


