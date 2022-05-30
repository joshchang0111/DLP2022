import ipdb
from torch.utils.data import DataLoader

from data.dataset import iclevrDataset

def load_train_data(args, device):
	"""Load i-CLEVR Dataset"""
	print("\nBuilding training & testing dataset...")

	train_dataset = iclevrDataset(args, device, "train")
	test_dataset  = iclevrDataset(args, device, "test")

	print("# training samples: {}".format(len(train_dataset)))
	print("# testing  samples: {}".format(len(test_dataset)))

	train_loader = DataLoader(
		train_dataset, 
		num_workers=args.num_workers, 
		batch_size=args.batch_size, 
		shuffle=True
	)
	test_loader = DataLoader(
		test_dataset, 
		num_workers=args.num_workers, 
		batch_size=args.batch_size, 
		shuffle=False
	)

	return train_loader, test_loader

def load_test_data(args, device):
	print("\nBuilding testing dataset...")

	test_dataset = iclevrDataset(args, device, "test")

	print("# testing  samples: {}".format(len(test_dataset)))

	test_loader = DataLoader(
		test_dataset, 
		num_workers=args.num_workers, 
		batch_size=args.batch_size, 
		shuffle=False
	)

	return test_loader