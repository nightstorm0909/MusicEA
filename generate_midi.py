#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:37:38 2020
"""
import sys, os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import pickle
import argparse
import numpy as np
from random import Random

import utilities.midi_utils as midi
from Individual import Individual

def init_flags_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--inputMidi', type=str, default=None, help='input midi file')
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

    if not args.inputMidi:
        raise Exception("Input midi file not spesified! Use -m <filename>")
    if not args.bestModel:
        raise Exception("Input model file not spesified! Use -b <filename>")
    if not args.saveName:
        raise Exception("Input name of outfile not spesified! Use -s <filename>")

    return args

def main(args=None):
    arguments = get_input(args)

     # get filenames
    model_filename = arguments.bestModel
    midi_filename = arguments.inputMidi
    new_filename = arguments.saveName

    # open old midi file
    csv_file = midi.midi_to_csv(midi_filename)
    notes = midi.csv_to_notes(csv_file)
    observed_sequence = notes.astype('int')

    with open(model_filename, 'rb') as f:
        stats = pickle.load(f)

    # plot statistics
    # stats.plot()

    best_individual = stats.bestIndividual

    genNotes, _ = best_individual.model.generate(len(notes))

    midi.notes_to_midi(midi_filename, genNotes, 'output/'+new_filename)

if __name__ == '__main__':
    main()
    # main(['-m', 'data/Never-Gonna-Give-You-Up-1.mid',
    #       '-b','output/stat_gen10_hid32_state7469.pickle',
    #       '-s', 'new_song_2'])
