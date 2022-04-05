# Lab3: 2048-Temporal Difference Learning
First create a directory for saving your results and model checkpoints.
```
$ mkdir Lab3/result/$EXPERIMENT_NAME$
$ mkdir Lab3/checkpoints/$EXPERIMENT_NAME$
```
Compile the main code ``2048.cpp``.
```
$ cd Lab3/src
$ sh compile.sh
```
Train the model by running ``train.sh``.
```
$ sh train.sh
```
Test the results of your model.
```
$ sh demo.sh
```
Note that the format of the execution line for both `train.sh` and `demo.sh` is as follows. 
```
$ 2048_main/2048 $MODE$ $EXPERIMENT_NAME$/
```
You have to specify the arguments correctly before running the program. `$MODE$` can be `train` or `demo`, and `$EXPERIMENT_NAME` corresponds to the directory name you want to store the results and model checkpoints.