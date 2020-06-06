import copy
import math
import numpy as np
import utilities.utils as utils
from encoder_decoder import rnnModel
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
		self.encoder  = rnnModel(input_size = Individual.inputSize, hidden_size = Individual.hiddenSize, output_size = Individual.outputSize, n = 4)
		self.decoder  = rnnModel(input_size = Individual.inputSize, hidden_size = Individual.hiddenSize, output_size = Individual.outputSize, n = 4)

		self.encoder_state_dict = self.encoder.get_w()
		self.encoder_states = list(self.encoder_state_dict.values())
		self.encoder_keys = list(self.encoder_state_dict.keys())
		
		self.decoder_state_dict = self.decoder.get_w()
		self.decoder_states = list(self.decoder_state_dict.values())
		self.decoder_keys = list(self.decoder_state_dict.keys())
		
		self.fit = None
		self.learningRate = 1 / math.sqrt(2*len(self.encoder_keys))
		self.learningRate2 = 1 / math.sqrt(2*math.sqrt(len(self.encoder_keys)))

		# sigma for encoder
		self.encoder_sigma = {}
		for key in self.encoder_keys:
			self.encoder_sigma[key] = np.random.uniform(0.9, 0.1)

		# sigma for decoder
		self.decoder_sigma = {}
		for key in self.decoder_keys:
			self.decoder_sigma[key] = np.random.uniform(0.9, 0.1)

	def crossover(self, other):
		#print("="*100)
		#print("before")
		#print('self.state_dict:{}, other.state_dict[key]:{}'.format(self.state_dict, other.state_dict))
		# Crossover for encoder
		for key in self.encoder_keys:
			alpha = np.random.random()
			tmp = (self.encoder_state_dict[key]*alpha) + (other.encoder_state_dict[key]*(1-alpha))
			other.encoder_state_dict[key] = (self.encoder_state_dict[key]*(1-alpha)) + (other.encoder_state_dict[key]*alpha)
			self.encoder_state_dict[key] = tmp
		
		self.encoder.set_w(self.encoder_state_dict)
		other.encoder.set_w(other.encoder_state_dict)
		#print("After")
		#print('self.state_dict:{}, other.state_dict:{}'.format(self.state_dict, other.state_dict))
		#print('self.model.state_dict:{}, other.model.state_dict:{}'.format(self.model.rnn.state_dict(), other.model.rnn.state_dict()))

		# Crossover for decoder
		for key in self.decoder_keys:
			alpha = np.random.random()
			tmp = (self.decoder_state_dict[key]*alpha) + (other.decoder_state_dict[key]*(1-alpha))
			other.decoder_state_dict[key] = (self.decoder_state_dict[key]*(1-alpha)) + (other.decoder_state_dict[key]*alpha)
			self.decoder_state_dict[key] = tmp
		
		self.decoder.set_w(self.encoder_state_dict)
		other.decoder.set_w(other.encoder_state_dict)

		self.fit = None
		other.fit = None
	
	def crossoverByTournament(self, other):
		child = copy.deepcopy(self)

		# crossover for encoder
		for key in self.encoder_keys:
			for i in range(len(self.encoder_state_dict[key])):
				if np.random.random() < 0.5:
					child.encoder_state_dict[key][i] = other.encoder_state_dict[key][i]

		# crossover for decoder
		for key in self.decoder_keys:
			for i in range(len(self.decoder_state_dict[key])):
				if np.random.random() < 0.5:
					child.decoder_state_dict[key][i] = other.decoder_state_dict[key][i]
		
		child.fit = None

		return child

	def mutate(self):
		tmp = self.learningRate * np.random.normal(0,1)
		#print("="*100)
		#print("before")
		#print('self.state_dict:{}, model.state_dict:{}'.format(self.state_dict, self.model.rnn.state_dict()))
		
		# Mutation for decoder.state_dict
		for key in self.encoder_keys:
			self.encoder_sigma[key] = self.encoder_sigma[key]*np.exp(tmp + self.learningRate2*np.random.normal(0,1))
			if self.encoder_sigma[key] < self.minSigma: self.encoder_sigma[key] = self.minSigma
			if self.encoder_sigma[key] > self.maxSigma: self.encoder_sigma[key] = self.maxSigma

			#for i, value in enumerate(self.state_dict[key]):
				#if np.random.random() < self.sigma[key]:
					#self.state_dict[key][i] = torch.randn(self.state_dict[key][i].shape)
					#self.state_dict[key][i] = np.random.uniform(-1, 1, self.state_dict[key][i].shape)
		
			if np.random.random() < self.encoder_sigma[key]:
					self.encoder_state_dict[key] = np.random.uniform(Individual.minLimit, Individual.maxLimit, self.encoder_state_dict[key].shape)

			#if self.x[i] > self.maxLimit: self.x[i]=self.maxLimit
			#if self.x[i] < self.minLimit: self.x[i]=self.minLimit
		#print("After")
		#print('self.state_dict:{}, model.state_dict:{}'.format(self.state_dict, self.model.rnn.state_dict()))

		self.encoder.set_w(self.encoder_state_dict)
		
		# Mutation for decoder.state_dict
		for key in self.decoder_keys:
			self.decoder_sigma[key] = self.decoder_sigma[key]*np.exp(tmp + self.learningRate2*np.random.normal(0,1))
			if self.decoder_sigma[key] < self.minSigma: self.decoder_sigma[key] = self.minSigma
			if self.decoder_sigma[key] > self.maxSigma: self.decoder_sigma[key] = self.maxSigma

			#for i, value in enumerate(self.state_dict[key]):
				#if np.random.random() < self.sigma[key]:
					#self.state_dict[key][i] = torch.randn(self.state_dict[key][i].shape)
					#self.state_dict[key][i] = np.random.uniform(-1, 1, self.state_dict[key][i].shape)
		
			if np.random.random() < self.decoder_sigma[key]:
					self.decoder_state_dict[key] = np.random.uniform(Individual.minLimit, Individual.maxLimit, self.decoder_state_dict[key].shape)

			#if self.x[i] > self.maxLimit: self.x[i]=self.maxLimit
			#if self.x[i] < self.minLimit: self.x[i]=self.minLimit
		#print("After")
		#print('self.state_dict:{}, model.state_dict:{}'.format(self.state_dict, self.model.rnn.state_dict()))

		self.decoder.set_w(self.decoder_state_dict)
		self.fit = None


	def evaluateFitness(self):
		if self.fit == None:
			self.embedding = self.encoder.encode(self.observed_sequence)
			decoded = self.decoder.decode(self.embedding, len(self.observed_sequence))
			decoded = decoded.reshape(-1)
			rmse = np.sqrt(np.sum((decoded - self.observed_sequence)**2))
			self.fit = -rmse 
			#print("fitness: ", self.fit)

	def generate(self, length, embedding = None):
		if embedding is None:
			generated = self.decoder.decode(self.embedding, length)
		else:
			generated = self.decoder.decode(embedding, length)
		return generated
	

	def __str__(self):
		return '[Individual Fit:{:6.5f}; Sigma:{}]'.format(self.fit, self.sigma)
