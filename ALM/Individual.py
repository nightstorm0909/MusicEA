#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 10:38:24 2020
Individual class for HMM model
"""

import numpy as np
import utilities.utils as utils
from ALM import AutoRegressiveLM as alm

class Individual:
	minSigma=1e-100
	maxSigma=1
	minLimit=None
	maxLimit=None
	N = None
	rand = None

	learningRate=None
	observed_sequence = None

	def __init__(self):
		self.model = alm(N = Individual.N, rand = Individual.rand)

		self.x = self.model.get_w()

		self.fit = None

		self.sigma = np.random.uniform(0.1,0.9)

	def crossover(self, other):
		alpha = np.random.random()

		# crossover for model.w
		tmp = (self.x*alpha) + (other.x*(1-alpha))
		other.x = (self.x*(1-alpha)) + (other.x*alpha)
		self.x = tmp
		
		self.model.set_w(self.x)
		other.model.set_w(other.x)
		self.fit = None
		other.fit = None

	def mutate(self):
		self.sigma=self.sigma*np.exp(self.learningRate*np.random.normal(0,1))
		if self.sigma < self.minSigma: self.sigma=self.minSigma
		if self.sigma > self.maxSigma: self.sigma=self.maxSigma

		# Mutation for model.w
		for i in range(len(self.x)):
			self.x[i] = self.x[i] + (self.maxLimit-self.minLimit)*self.sigma*np.random.normal(0,1)

		self.model.set_w(self.x)
		self.fit = None


	def evaluateFitness(self):
		if self.fit == None:
			self.fit = self.model.fitness(self.observed_sequence)
			#print("fitness: ", self.fit)

	def __str__(self):
		return '[Individual: {}:: . Fit:{:6.3f}; Sigma:{:6.3f}]'.format(self.x, self.fit, self.sigma)
