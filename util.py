# util.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import sys
import inspect
import heapq, random
# for load mat file and squeeze
import pickle
import numpy as np
from numpy import std, sqrt

def normalize(vec):
  sumOfMass = sum(vec)
  # if we have a zero vector, simply return it
  if sumOfMass == 0: return vec
  else: return map(lambda _: _ / sumOfMass, vec)

def standardErr(data):
  return std(data) / sqrt(len(data))

def powerset(iterable, minimum=0, maximum=None):
  from itertools import combinations
  s = list(iterable)
  if maximum is None: maximum = len(s)
  for r in range(minimum, maximum + 1):
    for _ in combinations(s, r):
      yield _

def sampleSubset(elems, subsetSize):
  """
  need this function because np.random.choice only works with 1-d array
  """
  selectedIndices = np.random.choice(range(len(elems)), subsetSize, replace=False)
  return [elems[idx] for idx in selectedIndices]

def getValueDistance(w1, w2):
  """
    Return:
      ||w1 - w2||_2
  """
  return np.linalg.norm([w1[key] - w2[key] for key in w1.keys()], np.inf)

def getMSE(x, y):
  assert len(x) == len(y)
  
  return sum([(xi - yi) ** 2 for xi, yi in zip(x, y)]) / len(x)

def checkPolicyConsistency(states, a, b):
  """
    Check how many policies on the states are consistent with the optimal one.

    Args:
      states: the set of states that we want to compare the policies
      a, b: two algorithms that we want to compare their policies
    Return:
      Portion of consistent policies
  """
  consistentPolices = 0

  # Walk through each state
  for state in states:
    consistentPolices += int(a.getPolicy(state) == b.getPolicy(state))

  return 1.0 * consistentPolices / len(states)

