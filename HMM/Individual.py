#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 10:38:24 2020
Individual class for HMM model
"""

import math
import numpy as np
import utilities.utils as utils
from utilities.HMM import HiddenMarkovModel as HMM

class Individual:
    minSigma=1e-100
    maxSigma=1

    learningRate=None
    nHiddenStates = None

    observed_sequence = None
    observed_states = None
    hidden_states = None

    def __init__(self):
        self._generate_hidden_states()

        self.model = HMM(self.hidden_states, self.observed_states)

        self.x = [self.model.get_pi(), self.model.get_A(), self.model.get_B()]

        self.fit = None

        self.sigma = [np.random.uniform(0.1,0.9) for _ in range(3)]


    def _generate_hidden_states(self):
        self.hidden_states = [i for i in range(self.nHiddenStates)]

    def crossover(self, other):
        #randomly choose position for model.pi
        pos0 = np.random.choice(range(len(self.x[0])))
        #randomly choose position for model.A
        pos1 = np.random.choice(range(self.x[1].shape[0]))
        #randomly choose position for model.B
        pos2 = np.random.choice(range(self.x[2].shape[0]))

        # crossover for model.pi
        tmp = np.copy(other.x[0][pos0:])
        other.x[0][pos0:] = self.x[0][pos0:]
        self.x[0][pos0:] = tmp
        self.x[0] = utils.normalize(self.x[0])
        other.x[0] = utils.normalize(other.x[0])
        self.model.set_pi(self.x[0])
        other.model.set_pi(other.x[0])

        # Crossover for model.A
        tmp = np.copy(other.x[1][pos1:])
        other.x[1][pos1:] = self.x[1][pos1:]
        self.x[1][pos1:] = tmp
        self.model.set_A(self.x[1])
        other.model.set_A(other.x[1])

        # Crossover for model.A
        tmp = np.copy(other.x[2][pos2:])
        other.x[2][pos2:] = self.x[2][pos2:]
        self.x[2][pos2:] = tmp
        self.model.set_B(self.x[2])
        other.model.set_B(other.x[2])

        self.fit = None
        other.fit = None

    def mutate(self):
        for i in range(len(self.sigma)):
            self.sigma[i] = self.sigma[i] * math.exp(self.learningRate * np.random.normal(0,1))
            if self.sigma[i] < self.minSigma: self.sigma[i] = self.minSigma
            if self.sigma[i] > self.maxSigma: self.sigma[i] = self.maxSigma

        # Mutation for model.pi
        if np.random.uniform() < self.sigma[0]:
            self.x[0] = utils.generate_prob_vector(len(self.x[0]))
            self.model.set_pi(self.x[0])

        # Mutation for model.A
        for i in range(self.x[1].shape[0]):
            if np.random.uniform() < self.sigma[1]:
                self.x[1][i] = utils.generate_prob_vector(len(self.x[1][i]))
        self.model.set_A(self.x[1])

        # Mutation for model.B
        for i in range(self.x[2].shape[0]):
            if np.random.uniform() < self.sigma[2]:
                self.x[2][i] = utils.generate_prob_vector(len(self.x[2][i]))
        self.model.set_B(self.x[2])

        self.fit = None


    def evaluateFitness(self):
        if self.fit == None:
            self.fit = self.model.score(self.observed_sequence)

    def __str__(self):
        return '[Individual. Fit:{:6.3f}; Sigmas:{:6.3f}-{:6.3f}-{:6.3f}]'.format(
            self.fit, self.sigma[0],self.sigma[1],self.sigma[2])
