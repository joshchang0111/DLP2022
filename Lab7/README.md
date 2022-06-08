# Lab7: Let's play GANs
## Dependencies
```
ipdb
tqdm
numpy
Pillow
torch==1.6.0+cu101
torchvision==0.7.0+cu101
```
Install pytorch as follows.
```
$ pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
## Dataset
Download the complete dataset `iclevr.zip` from the [link](https://drive.google.com/file/d/1y1x5aZjQR31IHKnXqRSqpNbOdDvIxXRc/view). Place it at your root data path, and remember to specify the path at the argument `--data_root` when running the code.
## Usage
Check out the commands written in `src/script.sh`.