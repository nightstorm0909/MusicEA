import sys
import os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import numpy as np
import pandas as pd
from itertools import product
from functools import reduce

import torch
import torch.nn as nn

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size

		self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
		self.h2o = nn.Linear(hidden_size, output_size)
		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()

	def forward(self, input, hidden):
		combined = torch.cat((input, hidden), 1)
		hidden = self.i2h(combined)
		hidden = self.sigmoid(hidden)
		#output = self.i2o(combined)
		output = self.h2o(hidden)
		output = self.tanh(output)
		return output, hidden
	
	def initHidden(self):
		return torch.zeros(1, self.hidden_size)

	def setWeights(self, state):
		pass

class RNN_numpy:
	def __init__(self, input_size, hidden_size, output_size):
		self.hidden_size = hidden_size
		self.input_size = input_size
		self.output_size = output_size

		self.state = {}
		self.state['i2h_weights'] = np.random.uniform(-1, 1, (self.input_size + self.hidden_size, self.hidden_size))
		self.state['i2h_bias'] = np.random.uniform(-1, 1, self.hidden_size)
		self.state['h2o_weights'] = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
		self.state['h2o_bias'] = np.random.uniform(-1, 1, self.output_size)

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def initHidden(self):
		return np.zeros((1, self.hidden_size))

	def forward(self, x, hidden):
		combined = np.concatenate((x, hidden), axis = 1)

		hidden = np.matmul(combined, self.state['i2h_weights']) + self.state['i2h_bias']
		hidden = self.sigmoid(hidden)

		output = np.matmul(hidden, self.state['h2o_weights']) + self.state['h2o_bias']
		output = np.tanh(output)

		return output, hidden

	def state_dict(self):
		return self.state

	def load_state_dict(self, values_dict):
		for key in values_dict.keys():
			if key in self.state:
				assert self.state[key].shape == values_dict[key].shape
				self.state[key] = values_dict[key]

			else:
				print("Wrong state_dict given for loading")
				exit(1)














