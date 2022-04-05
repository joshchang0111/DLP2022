import ipdb
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def read_bci_data(args):
    """Read data from preprocessed dataset."""
    print("Loading preprocessed data...")
    S4b_train  = np.load("{}/S4b_train.npz".format(args.data_path))
    X11b_train = np.load("{}/X11b_train.npz".format(args.data_path))
    S4b_test   = np.load("{}/S4b_test.npz".format(args.data_path))
    X11b_test  = np.load("{}/X11b_test.npz".format(args.data_path))

    train_data  = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data   = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label  = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label  = test_label - 1
    train_data  = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data   = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print("train data  shape: {}".format(train_data.shape))
    print("train label shape: {}".format(train_label.shape)) 
    print("test  data  shape: {}".format(test_data.shape))
    print("test  label shape: {}\n".format(test_label.shape))

    return train_data, train_label, test_data, test_label

def create_dataset(args, device, train_data, train_label, test_data, test_label):
    """Create batch dataset."""
    print("Creating dataset...\n")
    ## Convert numpy array to torch.Tensor, move data to GPU
    x_train = torch.Tensor(train_data).to(device)
    x_test  = torch.Tensor(test_data).to(device)
    y_train = torch.LongTensor(train_label).to(device)
    y_test  = torch.LongTensor(test_label).to(device)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset  = TensorDataset(x_test , y_test)

    ## Create batch dataset
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True)
    test_loader  = DataLoader(dataset=test_dataset , batch_size=args.bs, shuffle=False)

    return train_loader, test_loader

def save_data(args, data):
    """Receive a single sample and save its plot."""
    x = np.array(range(len(data[0][0])))

    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.plot(x, data[0][0])
    plt.ylabel("Channel 1")

    plt.subplot(2, 1, 2)
    plt.plot(x, data[0][1])
    plt.ylabel("Channel 2")

    plt.savefig("{}/sample.png".format(args.data_path))