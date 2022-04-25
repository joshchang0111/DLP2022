import pdb
import numpy as np
import pandas as pd

from PIL import Image

from torch.utils import data
from torchvision import transforms

def create_dataset(args):
    print("Build dataset...")

    train_dataset = RetinopathyLoader(root=args.data_path, mode="train")
    test_dataset  = RetinopathyLoader(root=args.data_path, mode="test")

    if not args.loss_weight:
        loss_weight = None
    else:
        loss_weight = train_dataset.loss_weight

    ## Create batch dataset
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True)
    test_loader  = data.DataLoader(dataset=test_dataset , batch_size=args.bs, shuffle=False)
    
    return train_dataset.num_classes, loss_weight, train_loader, test_loader

def getData(root, mode):
    if mode == "train":
        img   = pd.read_csv("{}/train_img.csv".format(root))
        label = pd.read_csv("{}/train_label.csv".format(root))
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img   = pd.read_csv("{}/test_img.csv".format(root))
        label = pd.read_csv("{}/test_label.csv".format(root))
        return np.squeeze(img.values), np.squeeze(label.values)

class RetinopathyLoader(data.Dataset):
    """Customized Dataset for retinopathy dataset"""
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.mode = mode
        self.img_name, self.label = getData(self.root, self.mode)
        print("{:5s} > Found {:5d} images...".format(self.mode, len(self.img_name)))

        ## Label infos
        self.classes = list(set(self.label))
        self.num_classes = len(self.classes)
        self.num_each_class = {}
        for class_ in self.classes:
            self.num_each_class[class_] = (self.label == class_).sum()
        print("{:7s} Number of each class: {}".format("", self.num_each_class))

        ## Weight of each class for loss function
        num_each_class = list(self.num_each_class.values())
        self.loss_weight = [max(num_each_class) / num for num in num_each_class]

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """

        img_path = "{}/data/{}.jpeg".format(self.root, self.img_name[index])
        label_gt = self.label[index]
        
        ## Define transformations
        transformations = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomVerticalFlip(), 
            transforms.ToTensor(), ## Scales the pixel to the range [0, 1]
            transforms.Normalize((0.3749, 0.2602, 0.1857),(0.2526, 0.1780, 0.1291)) ## Normalization
        ])

        img = Image.open(img_path)
        img = transformations(img)

        return img, label_gt
