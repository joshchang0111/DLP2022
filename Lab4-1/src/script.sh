python main.py --model EEGNet --save_plot
python main.py --model DeepConvNet --save_plot

################
## Experiment ##
################
#for lr in 5e-3 2e-3 1e-3 5e-4 2e-4 1e-4
#do
#	python main.py --model EEGNet --lr "$lr" --save_plot
#	python main.py --model DeepConvNet --lr "$lr" --save_plot
#done

## with dropout ##
#for i in $(seq 1 10)
#do
#	python main.py --model EEGNet
#	python main.py --model DeepConvNet
#done

## w/o dropout ##
#for i in $(seq 1 10)
#do
#	python main.py --model EEGNet --dropout 0
#	python main.py --model DeepConvNet --dropout 0
#done

######################
## Plot activations ##
######################
#python others/plot_act.py