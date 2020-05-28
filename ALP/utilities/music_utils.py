import sys, os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

def music_signal(filename: str) -> np.ndarray:
	'''
	Returns the music file as numpy array
	'''
	y, sr = librosa.load(filename)
	return y, sr

def music_save(filename: str, x: np.ndarray, sr: int):
	'''
	Saves the numpy array as wav file
	'''
	librosa.output.write_wav(os.path.join('output', "{}.wav".format(filename)), x, sr)
