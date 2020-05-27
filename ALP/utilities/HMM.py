#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 10:38:24 2020
"""
import sys
import os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import numpy as np
import pandas as pd
from itertools import product
from functools import reduce


class ProbabilityVector:
    def __init__(self, probabilities: dict):
        states = probabilities.keys()
        probs = list(probabilities.values())
        assert len(states) == len(probs), \
            "The probabilities must match the states."
        assert len(states) == len(set(states)), \
            "The states must be unique."
        assert abs(np.sum(probs) - 1.0) < 1e-12, \
            "Probabilities must sum up to 1."
        assert len(list(filter(lambda x: 0 <= x <= 1+1e-12, probs))) == len(probs), \
            "Probabilities must be numbers from [0, 1] interval."

        self.states = sorted(probabilities)
        self.values = np.array(list(map(lambda x:
            probabilities[x], self.states))).reshape(1, -1)

    @classmethod
    def initialize(cls, states: list):
        size = len(states)
        rand = np.random.rand(size) / (size**2) + 1 / size
        rand /= rand.sum(axis=0)
        return cls(dict(zip(states, rand)))

    @classmethod
    def from_numpy(cls, array: np.ndarray, states: list):
        return cls(dict(zip(states, list(array))))

    @property
    def dict(self):
        return {k:v for k, v in zip(self.states, list(self.values.flatten()))}

    @property
    def df(self):
        return pd.DataFrame(self.values, columns=self.states, index=['probability'])

    def __repr__(self):
        return "P({}) = {}.".format(self.states, self.values)

    def __eq__(self, other):
        if not isinstance(other, ProbabilityVector):
            raise NotImplementedError
        if (self.states == other.states) and (self.values == other.values).all():
            return True
        return False

    def __getitem__(self, state: str) -> float:
        if state not in self.states:
            raise ValueError("Requesting unknown probability state from vector.")
        index = self.states.index(state)
        return float(self.values[0, index])

    def __mul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityVector):
            return self.values * other.values
        elif isinstance(other, (int, float)):
            return self.values * other
        else:
            NotImplementedError

    def __rmul__(self, other) -> np.ndarray:
        return self.__mul__(other)

    def __matmul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityMatrix):
            return self.values @ other.values

    def __truediv__(self, number) -> np.ndarray:
        if not isinstance(number, (int, float)):
            raise NotImplementedError
        x = self.values
        return x / number if number != 0 else x / (number + 1e-12)

    def argmax(self):
        index = self.values.argmax()
        return self.states[index]

class ProbabilityMatrix:
    def __init__(self, prob_vec_dict: dict):

        assert len(prob_vec_dict) > 1, \
            "The numebr of input probability vector must be greater than one."
        assert len(set([str(x.states) for x in prob_vec_dict.values()])) == 1, \
            "All internal states of all the vectors must be indentical."
        assert len(prob_vec_dict.keys()) == len(set(prob_vec_dict.keys())), \
            "All observables must be unique."

        self.states   = sorted(prob_vec_dict)
        self.observables = prob_vec_dict[self.states[0]].states
        self.values   = np.stack([prob_vec_dict[x].values for x in self.states]).squeeze()

    @classmethod
    def initialize(cls, states: list, observables: list):
        observables = np.array(observables)
        states = np.array(states)

        rand = np.random.rand(states.size, len(observables)) / (states.size**2) + 1 / states.size
        rand /= rand.sum(axis=1).reshape(-1, 1)

        aggr = [dict(zip(observables, rand[i, :])) for i in range(len(states))]

        pvec = [ProbabilityVector(x) for x in aggr]

        return cls(dict(zip(states, pvec)))

    @classmethod
    def from_numpy(cls, array: np.ndarray, states: list, observables: list):
        p_vecs = [ProbabilityVector(dict(zip(observables, x))) for x in array]
        return cls(dict(zip(states, p_vecs)))

    @property
    def dict(self):
        return self.df.to_dict()

    @property
    def df(self):
        return pd.DataFrame(self.values, columns=self.observables, index=self.states)

    def __repr__(self):
        return "PM {} states: {} -> obs: {}.".format(
            self.values.shape, self.states, self.observables)

    def __getitem__(self, observable: str) -> np.ndarray:
        if observable not in self.observables:
            raise ValueError("Requesting unknown probability observable from the matrix.")
        index = self.observables.index(observable)
        return self.values[:, index].reshape(-1, 1)

class HiddenMarkovChain:
    def __init__(self, T, E, pi):
        self.T = T  # transmission matrix A
        self.E = E  # emission matrix B
        self.pi = pi
        self.states = pi.states
        self.observables = E.observables

    def __repr__(self):
        return "HML states: {} -> observables: {}.".format(
            len(self.states), len(self.observables))

    @classmethod
    def initialize(cls, states: list, observables: list):
        T = ProbabilityMatrix.initialize(states, states)
        E = ProbabilityMatrix.initialize(states, observables) # !!! Here
        pi = ProbabilityVector.initialize(states)
        return cls(T, E, pi)

    def _create_all_chains(self, chain_length):
        return list(product(*(self.states,) * chain_length))

    def _alphas(self, observations: list) -> np.ndarray:
        self.scale_factor = np.zeros(len(observations))
        alphas = np.zeros((len(observations), len(self.states)))

        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        # Scale alpha 0
        self.scale_factor[0] = 1 / np.sum(alphas[0, :])
        alphas[0, :] = alphas[0, :] * self.scale_factor[0]

        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1)
                         @ self.T.values) * self.E[observations[t]].T
            # Scale alpha
            self.scale_factor[t] = 1 / np.sum(alphas[t, :])
            alphas[t, :] = alphas[t, :] * self.scale_factor[t]
        return alphas

    def score(self, observations: list) -> float:
        alphas = self._alphas(observations)
        return -1 * np.sum(np.log(self.scale_factor)) #float(alphas[-1].sum())

    def run(self, length: int, pi_state:list=None) -> (list, list):
        assert length >= 0, "The chain needs to be a non-negative number."
        s_history = [0] * (length + 1)
        o_history = [0] * (length + 1)

        if pi_state is None:
            prb = self.pi.values
        else:
            prb = pi_state

        obs = prb @ self.E.values
        s_history[0] = np.random.choice(self.states, p=prb.flatten())
        o_history[0] = np.random.choice(self.observables, p=obs.flatten())

        for t in range(1, length + 1):
            prb = prb @ self.T.values
            obs = prb @ self.E.values
            s_history[t] = np.random.choice(self.states, p=prb.flatten())
            o_history[t] = np.random.choice(self.observables, p=obs.flatten())

        # return observations and state historys
        return o_history, s_history

class HiddenMarkovModel:
    def __init__(self, states:list, observables:list):
        self.layer = HiddenMarkovChain.initialize(states, observables)
        self._score = 0

    def score(self, observations: list) -> float:
        return self.layer.score(observations)

    def generate(self, length:int, pi_state:list=None) -> (list,list):
        return self.layer.run(length, pi_state)

    def get_A(self):
        return self.layer.T.df.to_numpy().copy()
    def get_B(self):
        return self.layer.E.df.to_numpy().copy()
    def get_pi(self):
        return self.layer.pi.df.to_numpy()[0].copy()

    def set_A(self, A):
        self.layer.T = ProbabilityMatrix.from_numpy(A,
                  self.layer.T.states, self.layer.T.observables)

    def set_B(self, B):
        self.layer.E = ProbabilityMatrix.from_numpy(B,
                  self.layer.E.states, self.layer.E.observables)

    def set_pi(self, pi):
        self.layer.pi = ProbabilityVector.from_numpy(pi,
                  self.layer.pi.states)
