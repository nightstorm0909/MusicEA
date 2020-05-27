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
* crossoverFraction - witch fraction of population combine
* randomSeed - random seed
* learningRate - learning rate
* nHiddenStates - number of hidden states in HMM
* nObservableStates - total amount of observations
* useFullSequence - if true will use nObservableStates, else will use unique values in observed sequence

[Mark Stamp. HMM](https://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf)