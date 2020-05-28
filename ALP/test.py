import time
import numpy as np
import pandas as pd
import librosa
import librosa.display
import argparse
import matplotlib.pyplot as plt
from ALP import AutoRegressiveLP as alp
from pandas.plotting import autocorrelation_plot
from utilities.music_utils import music_signal, music_save

def init_flags_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--input', type=str, default=None, help='input midi file')
	parser.add_argument('-b', '--bestModel', type=str, default=None, help='best model parameters')
	parser.add_argument('-s', '--saveName', type=str, default=None, help='generated midi file name')
	parser.add_argument('-q', '--quiet', action = "store_true", default = False, help='quiet mode')
	return parser

args = init_flags_parser()
args = args.parse_args()
print(args)

plt.figure()
y, sr = music_signal(args.input)
'''auto_cor = librosa.autocorrelate(y)
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

#model  = alp(np.array([1,1,2]))
model  = alp()
gen = model.generate(len(y[:512]))
s = time.time()
print("fitness:{}, in {} mins".format(model.fitness(y[:500000]), (time.time() - s) / 60))
music_save("test", y[:500000], sr)
