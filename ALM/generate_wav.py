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
from utilities.ev_config import EV_Config
from utilities.music_utils import music_signal, music_save

def init_flags_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c','--inputConfig', type=str, default=None, help='configuration file')
	parser.add_argument('-m', '--inputMusic', type=str, default=None, help='input midi file')
	parser.add_argument('-b', '--bestModel', type=str, default=None, help='best model parameters')
	parser.add_argument('-s', '--saveName', type=str, default=None, help='generated midi file name')
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

	gen = best_individual.model.generate(len(y))
	print(gen.shape)
	
	# plot original and generated music
	plt.figure()
	plt.subplots_adjust(hspace = 0.5)
	plt.subplot(211)
	plt.plot(y)
	plt.title('Original signal')
	plt.subplot(212)
	plt.plot(gen)
	plt.title('Generated Signal')
	plt.show()
	
	music_save(new_filename, gen, sr)

if __name__ == '__main__':
	main()
	# main(['-m', 'data/Never-Gonna-Give-You-Up-1.mid',
	#	   '-b','output/stat_gen10_hid32_state7469.pickle',
	#	   '-s', 'new_song_2'])
