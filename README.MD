# Evolutionary Hidden Markov models (HMM)
For population training, run:
```sh
python main.py -c config.cfg -m song.mid
```
For new midi file generation, run:
```sh
python generate_midi.py -b model.pickle -m song.mid -s new_song
```
Config file
* populationSize - size of population
* generationCount - number of generations
* crossoverFraction - which fraction of population combine
* randomSeed - random seed
* learningRate - learning rate
* nHiddenStates - number of hidden states in HMM
* nObservableStates - total amount of observations
* useFullSequence - if true will use nObservableStates, else will use unique values in observed sequence

[Mark Stamp. HMM](https://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf)

# Music generation using Autoregressive Linear Model(ALM)
For population training, run:
```sh
python main.py -c config.cfg -m train.mp3
```

# Evolutionary AutoRegressive Linear Model (ALM)
For population training, run:
```sh
cd ALM && python main.py -c config.cfg -m data/train.npy
```
For new music file generation, run:
```sh
python generate_wav.py -c config -b output/stat_gen2000_pop50_n50_False.pickle -m data/train.npy -s new_song
```
Config file
* populationSize - size of the population 
* generationCount - number of generations
* crossoverFraction - which fraction of population to combine
* randomSeed - random seed
* minLimit - minimum value of parameters of ALM
* maxLimit - maximum value of parameters of ALM
* samplingRate - new sampling rate for the music
* reduceLength - reducing the numpy array by 1/(reduceLength) fraction
* n - number of parameters of ALM
* random - whether to use randomness in the ALM
* multiSigma - whether to use mutiple sigmas for mutation
* generateFitness - fitness calculation by generating signal without any training input
* windowSize - window size (optional)
