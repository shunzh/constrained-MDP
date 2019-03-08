import random
import numpy as np
import itertools
import copy

class SimpleMDP:
  """
  An MDP object. All fields are initialized in __init__.
  """
  def __init__(self, S=[], A=[], T=None, r=None, alpha=None, terminal=lambda _: False, gamma=1, psi=[1]):
    self.S = S
    self.A = A
    self.T = T
    self.r = r
    self.alpha = alpha
    self.terminal = terminal
    self.gamma = gamma
    self.psi = psi
  
  def resetInitialState(self, initS):
    """
    reset the initial state distribution to be deterministically starting from initS
    """
    self.alpha = lambda s: s == initS


def constructDeterministicFactoredMDP(sSets, aSets, rFunc, tFunc, s0, gamma=1, terminal=lambda s: False):
  mdp = SimpleMDP()

  mdp.A = aSets

  # factored reward function
  #mdp.r = lambda state, action: sum(r(s, a) for s, r in zip(state, rFunc))
  # nonfactored reward function
  mdp.r = rFunc

  # transit(s, a) -> s'
  # the i-th component of s' is determined by tFunc[i]
  transit = lambda state, action: tuple([t(state, action) for t in tFunc])
  # transFunc(s, a, sp) -> prob
  # transFunc(s, a) -> s'
  #FIXME is this overriding used?
  def transFunc(state, action, sp=None):
    if sp == None:
      return transit(state, action)
    else:
      return 1 if sp == transit(state, action) else 0
  mdp.T = transFunc

  mdp.alpha = lambda s: s == s0 # assuming there is only one starting state

  mdp.terminal = terminal

  mdp.gamma = gamma

  # construct the set of reachable states
  mdp.S = []
  buffer = [s0]
  # stop when no new states are found by one-step transitions
  while len(buffer) > 0:
    # add the last batch to S
    mdp.S += buffer
    newBuffer = []
    for s in buffer:
      if not terminal(s):
        for a in aSets:
          sp = transit(s, a)
          if not sp in mdp.S and not sp in newBuffer:
            newBuffer.append(sp)
    buffer = newBuffer

  return mdp
