import copy
import math
import numpy as np
import utilities.utils as utils
from rnn import rnnModel
import torch

class Individual:
	minSigma=1e-100
	maxSigma=1
	minLimit=None
	maxLimit=None
	inputSize = None
	hiddenSize = None
	outputSize = None
	rand = None
	observed_sequence = None

	def __init__(self):
		self.model  = rnnModel(input_size = Individual.inputSize, hidden_size = Individual.hiddenSize, output_size = Individual.outputSize)

		self.state_dict = self.model.get_w()
		self.states = list(self.state_dict.values())
		self.keys = list(self.state_dict.keys())
		
		self.fit = None
		self.learningRate = 1 / math.sqrt(2*len(self.keys))
		self.learningRate2 = 1 / math.sqrt(2*math.sqrt(len(self.keys)))

		#self.sigma = [np.random.uniform(0.9,0.1) for _ in range(len(self.keys))]
		self.sigma = {}
		for key in self.keys:
			self.sigma[key] = np.random.uniform(0.9, 0.1)

	def crossover(self, other):
		#print("="*100)
		#print("before")
		#print('self.state_dict:{}, other.state_dict[key]:{}'.format(self.state_dict, other.state_dict))
		for key in self.keys:
			alpha = np.random.random()
			tmp = (self.state_dict[key]*alpha) + (other.state_dict[key]*(1-alpha))
			other.state_dict[key] = (self.state_dict[key]*(1-alpha)) + (other.state_dict[key]*alpha)
			self.state_dict[key] = tmp
		
		self.model.set_w(self.state_dict)
		other.model.set_w(other.state_dict)
		#print("After")
		#print('self.state_dict:{}, other.state_dict:{}'.format(self.state_dict, other.state_dict))
		#print('self.model.state_dict:{}, other.model.state_dict:{}'.format(self.model.rnn.state_dict(), other.model.rnn.state_dict()))

		self.fit = None
		other.fit = None
	
	def crossoverByTournament(self, other):
		child = copy.deepcopy(self)
		for key in self.keys:
			for i in range(len(self.state_dict[key])):
				if np.random.random() < 0.5:
					child.state_dict[key][i] = other.state_dict[key][i]
		return child

	def mutate(self):
		tmp = self.learningRate * np.random.normal(0,1)
		#print("="*100)
		#print("before")
		#print('self.state_dict:{}, model.state_dict:{}'.format(self.state_dict, self.model.rnn.state_dict()))
		
		# Mutation for model.state_dict
		for key in self.keys:
			self.sigma[key] = self.sigma[key]*np.exp(tmp + self.learningRate2*np.random.normal(0,1))
			if self.sigma[key] < self.minSigma: self.sigma[key] = self.minSigma
			if self.sigma[key] > self.maxSigma: self.sigma[key] = self.maxSigma

			#for i, value in enumerate(self.state_dict[key]):
				#if np.random.random() < self.sigma[key]:
					#self.state_dict[key][i] = torch.randn(self.state_dict[key][i].shape)
					#self.state_dict[key][i] = np.random.uniform(-1, 1, self.state_dict[key][i].shape)
		
			if np.random.random() < self.sigma[key]:
					self.state_dict[key] = np.random.uniform(Individual.minLimit, Individual.maxLimit, self.state_dict[key].shape)

			#if self.x[i] > self.maxLimit: self.x[i]=self.maxLimit
			#if self.x[i] < self.minLimit: self.x[i]=self.minLimit
		#print("After")
		#print('self.state_dict:{}, model.state_dict:{}'.format(self.state_dict, self.model.rnn.state_dict()))

		self.model.set_w(self.state_dict)
		self.fit = None


	def evaluateFitness(self):
		if self.fit == None:
			self.fit = self.model.fitness(self.observed_sequence)
			#print("fitness: ", self.fit)

	def __str__(self):
		return '[Individual Fit:{:6.5f}; Sigma:{}]'.format(self.fit, self.sigma)
