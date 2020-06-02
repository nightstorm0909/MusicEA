import sys, os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import pickle
import librosa
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean,stdev

#Print some useful stats to screen
def printStats(pop,gen):
	print('Generation:',gen)
	avgval=0
	maxval=pop[0].fit
	sigma=pop[0].sigma
	for ind in pop:
		avgval+=ind.fit
		if ind.fit > maxval:
			maxval=ind.fit
			sigma=ind.sigma
		#print(ind)

	print('Max fitness',maxval)
	print('Sigma',sigma)
	print('Avg fitness',avgval/len(pop))
	print('')

def normalize(x):
	return x / x.sum(0)

def save_model(model, filename):
	with open(filename, 'wb') as f:
		pickle.dump(model, f)

#Basic class for computing, managing, plotting EV3 run stats
class EV_Stats:
	def __init__(self):
		self.bestIndividual = []
		self.bestFit=[]
		self.meanFit=[]
		self.stddevFit=[]
		self.mutationStrength = []
		self.finished_time = 0.0

	def accumulate(self, pop):
		#find state with max fitness
		max_fit = pop[0].fit
		max_fit_individ = pop[0]
		mut_strength = pop[0].sigma

		for p in pop:
			if p.fit > max_fit:
				 max_fit = p.fit
				 max_fit_individ = p
				 mut_strength = p.sigma

		self.bestIndividual.append(max_fit_individ)
		self.bestFit.append(max_fit)
		self.mutationStrength.append(mut_strength)

		#compute mean and stddev
		fits = [p.fit for p in pop]
		self.meanFit.append(mean(fits))
		self.stddevFit.append(stdev(fits))

	def print(self, gen=None):
		#if gen not specified, print latest
		if gen is None: gen = len(self.bestFit)-1
		print('Generation:',gen)
		print('Best Individual: ', self.bestIndividual[gen].x)
		print('Best fitness	 : {:6.3f}'.format(self.bestFit[gen]))
		print('Mean fitness	 : {:6.3f}'.format(self.meanFit[gen]))
		print('Stddev fitness   : {}'.format(self.stddevFit[gen]))
		print('Mutation Strength: {}'.format(str(self.mutationStrength[gen])))

	def plot(self):
		#plot stats to screen & file using matplotlib
		gens = range(len(self.bestFit))

		plt.figure()
		#create stacked plots (4x1)
		plt.subplots_adjust(hspace=0.5)
		plt.subplot(411)
		plt.plot(gens,self.bestFit)
		plt.ylabel('Max Fit')
		plt.title('EV Run Statistics')
		plt.subplot(412)
		plt.plot(gens,self.meanFit)
		plt.ylabel('Mean Fit')
		plt.subplot(413)
		plt.plot(gens,self.stddevFit)
		plt.ylabel('Stddev Fit')
		plt.subplot(414)
		plt.plot(gens,self.mutationStrength)
		plt.ylabel('Mutation Strength')

		#write plots to .png file, then display to screen
		plt.savefig('output/statistics.png')
		plt.show()

	def __str__(self):
		s=''
		s+='bestFits  : ' + str(self.bestFit) + '\n'
		#s+='bestStates: ' + str(self.bestState) + '\n'
		s+='meanFits  : ' + str(self.meanFit) + '\n'
		s+='stddevFits: ' + str(self.stddevFit) + '\n'
		s+=':mutationStrength ' + str(self.mutationStrength)
		return s
