import sys, os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import pickle
import librosa
import argparse
import numpy as np
from random import Random
import matplotlib.pyplot as plt
from Individual import Individual
from rnn import rnnModel
from utilities.ev_config import EV_Config
from utilities.music_utils import music_signal, music_save

def init_flags_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c','--inputConfig', type=str, default=None, help='configuration file')
	parser.add_argument('-m', '--inputMusic', type=str, default=None, help='input music file')
	parser.add_argument('-b', '--bestModel', type=str, default=None, help='best model parameters')
	parser.add_argument('-s', '--saveName', type=str, default=None, help='generated music file name')
	parser.add_argument('-q', '--quiet', action = "store_true", default = False, help='quiet mode')
	return parser


def get_input(args):
	parser = init_flags_parser()

	if isinstance(args, list):
		# Parse args passed to the main function
		args = parser.parse_args(args)
	else:
		# Parse args from terminal
		args = parser.parse_args()

	if not args.inputConfig:
		raise Exception("Input config file not spesified! Use -c <filename>")
	if not args.inputMusic:
		raise Exception("Input music file not spesified! Use -m <filename>")
	if not args.bestModel:
		raise Exception("Input model file not spesified! Use -b <filename>")
	if not args.saveName:
		raise Exception("Input name of outfile not spesified! Use -s <filename>")

	return args

def main(args=None):
	arguments = get_input(args)

	# get filenames
	config_filename = arguments.inputConfig
	model_filename = arguments.bestModel
	music_filename = arguments.inputMusic
	new_filename = arguments.saveName

	# get EV3 config parameters
	config = EV_Config(os.path.join(path, config_filename))
	
	# open old music file
	y, sr = music_signal(music_filename)
	sr2 = config.samplingRate
	y = librosa.core.resample(y, sr, sr2)
	y = y[:int(len(y)/config.reduceLength)] 

	with open(model_filename, 'rb') as f:
		stats = pickle.load(f)

	# plot statistics
	stats.plot()

	best_individual = stats.bestIndividual[-1]

	# Generate whole music
	print(best_individual.model.get_w()['i2h_weights'].shape, best_individual.model.get_w()['h2o_weights'].shape)
	hidden_size = best_individual.model.get_w()['i2h_weights'].shape[1]
	input_size = best_individual.model.get_w()['i2h_weights'].shape[0] - hidden_size
	output_size = input_size

	model = rnnModel(	input_size = input_size,
							hidden_size = hidden_size,
							output_size = output_size,
							n = len(best_individual.model.get_w().keys()))
	print(model.get_w().keys())
	print(best_individual.model.get_w().keys())
	model.set_w(best_individual.model.get_w())
	gen = model.generate(len(y))
	print(gen.shape)

	# Complete the music
	input_music_length = 10000
	y_hat = y[:input_music_length]

	hidden = model.rnn.initHidden()
	for i in range(input_music_length):
		_, hidden = model.rnn.forward(y_hat[i].reshape(-1, 1), hidden)

	complete = []
	for i in range(len(y) - input_music_length):
		if i > 0:
			output, hidden = model.rnn.forward(output, hidden)
		else:
			output, hidden = model.rnn.forward(y_hat[-1].reshape(-1, 1), hidden)
		complete.append(output)

	complete = np.array(complete)
	complete = complete.reshape(-1)
	print(complete.shape, y_hat.shape)
	y_hat = np.concatenate((y_hat, complete))
	print(y_hat.shape)
	
	# plot original and generated music
	plt.figure()
	plt.subplots_adjust(hspace = 0.5)
	plt.subplot(311)
	plt.plot(y)
	plt.title('Original signal')
	plt.subplot(312)
	plt.plot(gen)
	plt.title('Generated Signal')
	plt.subplot(313)
	plt.plot(y_hat)
	plt.title('Completed signal')
	plt.show()
	
	# plot original and generated music
	plt.figure()
	plt.subplots_adjust(wspace = 0.5)
	plt.subplot(121)
	plt.hist(y, 100)
	plt.title('Music Histogram')
	plt.subplot(122)
	plt.hist(gen, 100)
	plt.title('RNN Generation Histogram')
	plt.show()
	
	#music_save("{}2".format(new_filename), y_hat, sr2)
	music_save(new_filename, gen, sr2)
	return stats, y, gen, y_hat

if __name__ == '__main__':
	stats, y, gen, y_hat = main()
	# main(['-m', 'data/Never-Gonna-Give-You-Up-1.mid',
	#	   '-b','output/stat_gen10_hid32_state7469.pickle',
	#	   '-s', 'new_song_2'])
