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
