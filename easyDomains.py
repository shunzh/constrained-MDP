import random
import numpy as np
import itertools
import copy

class SimpleMDP:
  def __init__(self, S=[], A=[], T=None, r=None, alpha=None, terminal=lambda _: False, gamma=1, psi=[1]):
    """
    Most methods do component assignment separately, so setting dummy default values.
    """
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


def getFactoredMDP(sSets, aSets, rFunc, tFunc, s0, gamma=1, terminal=lambda s: False):
  ret = SimpleMDP()

  ret.A = aSets
  # factored reward function
  #ret.r = lambda state, action: sum(r(s, a) for s, r in zip(state, rFunc))
  # nonfactored reward function
  ret.r = rFunc

  # t(s, a, s') = \prod t_i(s, a, s_i)
  transit = lambda state, action: tuple([t(state, action) for t in tFunc])
  
  # overriding this function depending on if sp is passed in
  #FIXME assume deterministic transitions for now to make the life easier!
  def transFunc(state, action, sp=None):
    if sp == None:
      return transit(state, action)
    else:
      return 1 if sp == transit(state, action) else 0

  ret.T = transFunc 
  ret.alpha = lambda s: s == s0 # assuming there is only one starting state
  ret.terminal = terminal
  ret.gamma = gamma

  #print transit(((2, 1), 0, 0, 1, 0, 1, 3), (1, 0))
  
  # construct the set of reachable states
  ret.S = []
  buffer = [s0]
  # stop when no new states are found by one-step transitions
  while len(buffer) > 0:
    # add the last batch to S
    ret.S += buffer
    newBuffer = []
    for s in buffer:
      if not terminal(s):
        for a in aSets:
          sp = transit(s, a)
          if not sp in ret.S and not sp in newBuffer: 
            newBuffer.append(sp)
    buffer = newBuffer

  return ret
