#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:37:38 2020
"""
import time
import pickle
import librosa
import numpy as np
import matplotlib.pyplot as plt
from Population import Population
from Individual import Individual
from multiprocessing import Pool
from ALM import AutoRegressiveLM as alm
from utilities.utils import EV_Stats, save_model
from utilities.music_utils import music_signal, music_save

class EV:
	def __init__(self, config, music_filename):
		# get sequence
		y, sr = music_signal(music_filename)
		self.sr = config.samplingRate
		self.observed_sequence = librosa.core.resample(y, sr, self.sr)
		self.observed_sequence =self.observed_sequence[:int(len(self.observed_sequence)/config.reduceLength)] 
		print("[INFO] resampled sequence length: ", self.observed_sequence.shape)

		# setup random number generator
		np.random.seed(config.randomSeed)

		# Individual initialization
		Individual.observed_sequence = self.observed_sequence
		#Individual.learningRate = config.learningRate
		Individual.minLimit=config.minLimit
		Individual.maxLimit=config.maxLimit
		Individual.N = config.n
		Individual.rand = config.random
		Individual.multi_dim_mut_rate = config.multiSigma

		# Population initialization
		Population.crossoverFraction = config.crossoverFraction
		Population.pool = Pool()

		# save sequence and config
		self.config = config

	def run(self):
		start = time.time()
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

		stats.finished_time = (time.time() - start) / 3600
		# plot accumulated stats to file/screen using matplotlib
		stats.plot()
		ind = stats.bestIndividual[-1]

		# save statistics
		f_name = "stat_gen{}_pop{}_n{}_rand{}_reduce{}.pickle".format(self.config.generationCount,
																						self.config.populationSize,
																						len(ind.model.w),
																						ind.rand,
																						self.config.reduceLength)

		save_model(stats, 'output/'+f_name)
		y_hat = ind.model.generate(len(self.observed_sequence))
		self.gen = y_hat
		#plt.hist(y_hat, 100)
		#plt.show()
		music_save("gen{}_pop{}_n{}_rand{}_reduce{}".format(self.config.generationCount,
																			self.config.populationSize,
																			len(ind.model.w),
																			ind.rand,
																			self.config.reduceLength), y_hat, self.sr)




