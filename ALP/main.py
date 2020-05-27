#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:37:38 2020
"""
import time, sys, os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import argparse
from EV import EV
from utilities.ev_config import EV_Config

def init_flags_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c','--inputConfig', type=str, default=None, help='configuration file')
	parser.add_argument('-m', '--inputMidi', type=str, default=None, help='midi file')
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
	if not args.inputMidi:
		raise Exception("Input midi file not spesified! Use -m <filename>")
	return args


def main(args=None):
	start_time = time.time()

    # get arguments from terminal
	arguments = get_input(args)

    # get filenames
	config_filename = arguments.inputConfig
	midi_filename = arguments.inputMidi

	# get EV3 config parameters
	config = EV_Config(os.path.join(path, config_filename))

	# print config parameters
	print("Config parameters:\n" + str(config))

 	# run evolution
	ev = EV(config, midi_filename)
	ev.run()

	print("[INFO] Finished in {:6.3f} hours".format((time.time() - start_time)/3600))

if __name__ == '__main__':
    main()
# 	main(['-c','config.cfg', '-m', 'data/Never-Gonna-Give-You-Up-1.mid'])
