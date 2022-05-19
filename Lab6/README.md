# Lab6: Deep Q-Network and Deep Deterministic Policy Gradient
## Dependencies
```
ipdb
gym
numpy
tensorboard
torch==1.6.0+cu101
```
Install pytorch as follows.
```
$ pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
Install Box2D as follows if you are using the same OS version as me.
```
$ sudo apt-get install swig build-essential python-dev python3-dev
$ pip install Box2D
$ pip install box2d-py
$ pip install gym[box2d]
```
## Usage
Run the following command to train and test DQN.
```
$ cd Lab6/src
$ python dqn.py --logdir ../log/dqn
```
Run the following command to traing and test DDPG.
```
$ python ddpg.py --logdir ../log/dqn
```