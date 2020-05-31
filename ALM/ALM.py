import random
import numpy as np

class AutoRegressiveLM:
	N = 3
	rand = None

	def __init__(self, w: np.ndarray = None):
		if w is None:
			#self.w = [random.random() for _ in range(AutoRegressiveLM.N)]
			self.w = np.random.uniform(-1, 1, AutoRegressiveLM.N)
		else:
			self.w = w

	def generate(self, length: int) -> np.ndarray:
		'''
		Returns a generated sequence according to the model
		'''
		history = np.zeros(self.w.shape[0])
		generated_seq = []
		for i in range(length):
			if self.rand:
				a0 = np.random.normal(0, 1, 1)
			else:
				a0 = 1
			history = np.insert(history, 0, a0)[:-1]
			res = np.sum(self.w * history)
			generated_seq.append(res)
			history[0] = res
			#print("generate: {}, history: {}, a0: {}, res: {}". format(generated_seq, history, a0, res))

		return np.array(generated_seq)

	def generate_with_history(self, history: np.ndarray) -> np.array:
		'''
		Returns a generated value according to the history provided
		'''
		if self.rand:
			a0 = np.random.normal(0, 1, 1)
		else:
			a0 = 1
		history = np.insert(history, 0, a0)[:-1]
		res = np.sum(self.w * history)
		return res

	def fitness(self, seq: np.ndarray) -> np.ndarray:
		'''
		Calculate the root mean square between generated sequence and given sequence
		'''
		#generated_seq = self.generate(len(seq))
		rmse = 0
		generated_seq = []
		history = np.zeros(self.w.shape[0])
		for i in range(len(seq)):
			generated_seq.append(self.generate_with_history(history))
			history = np.insert(history, 0, seq[i])[:-1]
			#print("generate: {}, history: {}, seq: {}". format(generated_seq, history, seq))

		rmse = np.sqrt(np.sum((seq - generated_seq)**2))
		return -rmse

	def __str__(self):
		return np.array2string(self.w)

	def get_w(self):
		'''
		Get the parameters of the model
		'''
		return self.w

	def set_w(self, value: np.array):
		'''
		Set the parameters of the model
		'''
		assert value.shape == self.w.shape
		self.w = value