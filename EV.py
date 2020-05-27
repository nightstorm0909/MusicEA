#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:37:38 2020
"""
import time
import pickle
import numpy as np
import utilities.midi_utils as midi
from Population import Population
from Individual import Individual
from multiprocessing import Pool
from utilities.utils import EV_Stats, save_model

class EV:
    def __init__(self, config, midi_filename):
        # get sequence
        csv_file = midi.midi_to_csv(midi_filename)
        notes = midi.csv_to_notes(csv_file)
        observed_sequence = notes.astype('int') #list(notes)

        # states used in the individuals
        if not config.useFullSequence:
            observed_states = list(np.unique(observed_sequence))
        else:
            observed_states = np.arange(config.nObservableStates)

        # setup random number generator
        np.random.seed(config.randomSeed)

        # Individual initialization
        Individual.observed_sequence = observed_sequence
        Individual.nHiddenStates = config.nHiddenStates
        Individual.learningRate = config.learningRate
        Individual.observed_states = observed_states

        # Population initialization
        Population.crossoverFraction = config.crossoverFraction
        Population.pool = Pool()

        # save sequence and config
        self.config = config
        self.observed_sequence = observed_sequence

    def run(self):
        # create initial Population (random initialization)
        population = Population(self.config.populationSize)
        population.evaluateFitness()

        # accumulate & print stats
        stats = EV_Stats()
        stats.accumulate(population)
        stats.print()
        print()

        # evolution main loop
        for i in range(self.config.generationCount):
            gen_start = time.time()

            #create initial offspring population by copying parent pop
            offsprings = population.copy()

            #select mating pool
            offsprings.conductTournament()

            #perform crossover
            offsprings.crossover()

            #random mutation
            offsprings.mutate()

            #update fitness values
            offsprings.evaluateFitness()

            #survivor selection: elitist truncation using parents+offspring
            population.combinePops(offsprings)
            population.truncateSelect(self.config.populationSize)

            # accumulate & print stats
            stats.accumulate(population)
            stats.print()
            print("[INFO] Generation {} finished in {:6.3f} minutes\n".format(
                i+1, (time.time() - gen_start)/60))

        # plot accumulated stats to file/screen using matplotlib
        stats.plot()

        # save statistics
        f_name = "stat_gen{}_hid{}_state{}.pickle".format(
            self.config.generationCount,
            self.config.nHiddenStates,
            len(self.observed_sequence))

        save_model(stats, 'output/'+f_name)
