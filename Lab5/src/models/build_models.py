import ipdb

## Self-defined
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_encoder, vgg_decoder
from others.utils import init_weights

def build_models(args, saved_model, device):
	print("\nBuilding models...")

	######################
	## Build the models ##
	######################
	if args.model_dir != "":
		frame_predictor = saved_model["frame_predictor"]
		posterior = saved_model["posterior"]
	else:
		#frame_predictor = lstm(args.g_dim + args.z_dim, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
		frame_predictor = lstm(args.g_dim + args.z_dim + args.cond_dim, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
		posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)
		frame_predictor.apply(init_weights)
		posterior.apply(init_weights)
			
	if args.model_dir != "":
		decoder = saved_model["decoder"]
		encoder = saved_model["encoder"]
	else:
		encoder = vgg_encoder(args.g_dim)
		decoder = vgg_decoder(args.g_dim)
		encoder.apply(init_weights)
		decoder.apply(init_weights)
	
	########################
	## Transfer to device ##
	########################
	frame_predictor.to(device)
	posterior.to(device)
	encoder.to(device)
	decoder.to(device)

	return frame_predictor, posterior, encoder, decoder