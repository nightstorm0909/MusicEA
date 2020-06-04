import time
import torch
import numpy as np
import pandas as pd
import librosa
import librosa.display
import argparse
import matplotlib.pyplot as plt
from rnn import rnnModel
from Population import Population
from Individual import Individual
from pandas.plotting import autocorrelation_plot
from utilities.music_utils import music_signal, music_save
from utilities.ev_config import EV_Config

def init_flags_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c','--inputConfig', type=str, default=None, help='configuration file')
	parser.add_argument('-m', '--input', type=str, default=None, help='input midi file')
	parser.add_argument('-b', '--bestModel', type=str, default=None, help='best model parameters')
	parser.add_argument('-s', '--saveName', type=str, default=None, help='generated midi file name')
	parser.add_argument('-q', '--quiet', action = "store_true", default = False, help='quiet mode')
	return parser

args = init_flags_parser()
args = args.parse_args()
print(args)
config = EV_Config(args.inputConfig)

y, sr = music_signal(args.input)
'''with open('train.npy', 'wb') as f:
np.save(f, np.array([y, sr]))
auto_cor = librosa.autocorrelate(y)
plt.subplot(3, 1, 1)
librosa.display.waveplot(y, sr=sr)
plt.subplot(3, 1, 2)
plt.plot(auto_cor)
plt.title('Auto-correlation')
plt.xlabel('Lag (frames)')
plt.subplot(3, 1, 3)
#autocorrelation_plot(pd.DataFrame(y)[:1024])
plt.show()
'''

# Individual initialization
#Individual.learningRate = config.learningRate
Individual.minLimit=config.minLimit
Individual.maxLimit=config.maxLimit
Individual.inputSize = config.inputSize
Individual.hiddenSize = config.hiddenSize
Individual.outputSize = config.outputSize

# Population initialization
Population.crossoverFraction = config.crossoverFraction
pop = Population(2)

model  = rnnModel(1, 10, 1)
model2  = rnnModel(1, 10, 1)
print(y.shape, sr)
s = time.time()
y_hat = librosa.core.resample(y, sr, 2000)
y_hat = y_hat[:int(len(y_hat)/6)]
print(y_hat.shape)
#gen = model.generate(len(y_hat), 0.01)
#fit = model.fitness2(y_hat)
#print("fitness:{}, in {} mins".format(model.fitness(y_hat), (time.time() - s) / 60))
#music_save("train", y_hat[:], 2000)
#y_hat2 = librosa.core.resample(y_hat, 1000, sr)
#music_save("test2", y_hat2, sr)
a = pop[0]
b = pop[1]
#a.crossover(b)
a.mutate()


