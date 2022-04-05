from models.EEGNet import EEGNet
from models.DeepConvNet import DeepConvNet

def build_model(args):
	print("Building model...\n")
	if args.model == "EEGNet":
		model = EEGNet(activation=args.activation, dropout_p=args.dropout)
	elif args.model == "DeepConvNet":
		model = DeepConvNet(activation=args.activation, dropout_p=args.dropout)
	else:
		raise NotImplementedError("Model {} not implemented!".format(args.model))
	return model
