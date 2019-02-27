'''Parser for model training from command line'''

import argparse
import json

def parse_args(config_file="configs_model.json"):

	# default file paths if not entered by user
	default_configs = json.load(open(config_file))

	up_fac = default_configs["upscale_factor"]
	tr_bs = default_configs["train_batch_size"]
	te_bs = default_configs["test_batch_size"]
	n_ep = default_configs["num_epochs"]
	lr = default_configs["learning_rate"]
	thr = default_configs["threads_data_loader"]
	seed = default_configs["random_seed"]

	parser = argparse.ArgumentParser()
	
	parser.add_argument('--upscale_factor', type = int, default = up_fac, \
		help = 'super resolution upscale factor')
	parser.add_argument('--train_batch_size', type = int, default = tr_bs, \
		help = 'training batch size')
	parser.add_argument('--test_batch_size', type = int, default = te_bs, \
		help = 'testing batch size')
	parser.add_argument('--num_epochs', type = int, default = n_ep, \
		help = 'number of epochs for training')
	parser.add_argument('--lr', type = float, default = lr, \
		help = 'learning rate')
	parser.add_argument('--threads', type = int, default = thr, \
		help = 'number of threads for dataloader to use')
	parser.add_argument('--seed', type = int, default = seed, \
		help = 'random seed to use')

	args = parser.parse_args()

	# TODO: Add error handling

	return args