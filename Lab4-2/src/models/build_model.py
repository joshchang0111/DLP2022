from models.resnet import resnet

def build_model(args, num_class):
	print("\nBuild model...")

	pretrained_str = "w/" if args.pretrained else "w/o"
	print("Using {}, {} pretrained parameters...\n".format(args.model, pretrained_str))

	model = resnet(args.model, args.pretrained, num_class)
	
	return model