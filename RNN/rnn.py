import random
import numpy as np
import warnings
import torch
#from utilities.RNN import RNN
from utilities.RNN import RNN_numpy as RNN

class rnnModel:
	def __init__(self, input_size:int, hidden_size: int, output_size: int, n: int = 6):
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		
		self.rnn = RNN(self.input_size, self.hidden_size, self.output_size, n)

	def generate(self, length: int, init: int = None, seq: list = None) -> np.ndarray:
		'''
		Returns a generated sequence according to the model
		'''
		generated_seq = []
		hidden = self.rnn.initHidden()

		if init is None:
			init = np.random.normal(1, 0.01)
		#init = torch.tensor(init).view(-1, 1)
		init = np.array(init).reshape(-1, 1)
		#print(init.shape, hidden.shape)
		for i in range(length + 1):
			if i > 0:
				if seq is None:
					#output, hidden = self.rnn(output, hidden)
					output, hidden = self.rnn.forward(output, hidden)
				else:
					#output, hidden = self.rnn(torch.tensor(seq[i-1]).view(-1, 1), hidden)
					output, hidden = self.rnn.forward(np.array(seq[i-1]).reshape(-1, 1), hidden)
			else:
				#output, hidden = self.rnn(init, hidden)
				output, hidden = self.rnn.forward(init, hidden)
			#generated_seq.append(output.detach().numpy()[0][0])
			generated_seq.append(output[0][0])
		
		#print(generated_seq[-20:])
		return np.array(generated_seq[1:])

	def fitness(self, seq: np.ndarray) -> np.ndarray:
		'''
		Calculate the root mean square between generated sequence with history and given sequence
		'''
		#generated_seq = self.generate(len(seq), np.random.normal(0, 1), seq)
		#generated_seq = self.generate(len(seq), np.random.normal(1, 0.01))
		generated_seq = self.generate(len(seq), 0.1)
		rmse = np.sqrt(np.sum((seq - generated_seq)**2))
		#print(rmse)
		return -rmse

	def __str__(self):
		#return np.array2string(self.w)
		pass

	def get_w(self):
		'''
		Get the parameters of the model
		'''
		return self.rnn.state_dict()

	def set_w(self, values: dict):
		'''
		Set the parameters of the model
		'''
		self.rnn.load_state_dict(values)













