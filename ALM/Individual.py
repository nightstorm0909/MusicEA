import math
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
	multi_dim_mut_rate = None
	observed_sequence = None
	windowSize = None

	def __init__(self):
		self.model = alm(N = Individual.N, rand = Individual.rand, window_size = Individual.windowSize)

		self.x = self.model.get_w()

		self.fit = None
		self.learningRate = 1 / math.sqrt(2*len(self.x))
		self.learningRate2 = 1 / math.sqrt(2*math.sqrt(len(self.x)))

		if Individual.multi_dim_mut_rate:
			self.sigma = [np.random.uniform(0.9,0.1) for _ in range(len(self.x))]
		else:
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
		if Individual.multi_dim_mut_rate:
			tmp = self.learningRate * np.random.normal(0,1)
			
			# Mutation for model.w
			for i in range(len(self.x)):
				self.sigma[i] = self.sigma[i]*np.exp(tmp + self.learningRate2*np.random.normal(0,1))
				if self.sigma[i] < self.minSigma: self.sigma[i] = self.minSigma
				if self.sigma[i] > self.maxSigma: self.sigma[i] = self.maxSigma
				self.x[i] = self.x[i] + (self.maxLimit-self.minLimit)*self.sigma[i]*np.random.normal(0,1)
			
				if self.x[i] > self.maxLimit: self.x[i]=self.maxLimit
				if self.x[i] < self.minLimit: self.x[i]=self.minLimit
		else:
			self.sigma=self.sigma*np.exp(self.learningRate*np.random.normal(0,1))
			if self.sigma < self.minSigma: self.sigma=self.minSigma
			if self.sigma > self.maxSigma: self.sigma=self.maxSigma

			# Mutation for model.w
			for i in range(len(self.x)):
				self.x[i] = self.x[i] + (self.maxLimit-self.minLimit)*self.sigma*np.random.normal(0,1)

				if self.x[i] > self.maxLimit: self.x[i]=self.maxLimit
				if self.x[i] < self.minLimit: self.x[i]=self.minLimit

		self.model.set_w(self.x)
		self.fit = None


	def evaluateFitness(self):
		if self.fit == None:
			self.fit = self.model.fitness(self.observed_sequence)
			#print("fitness: ", self.fit)

	def evaluateFitness2(self):
		if self.fit == None:
			self.fit = self.model.fitness2(self.observed_sequence)
			#print("fitness: ", self.fit)

	def evaluateFitnessWithWindow(self):
		if self.fit == None:
			self.fit = self.model.fitness_with_window(self.observed_sequence)
			#print("fitness: ", self.fit)
	
	def __str__(self):
		return '[Individual: {}:: . Fit:{:6.3f}; Sigma:{:6.3f}]'.format(self.x, self.fit, self.sigma)
