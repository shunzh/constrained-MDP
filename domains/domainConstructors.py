import copy

from util import normalize


class SimpleMDP:
  """
  An MDP object.
  Reward could be certain or uncertain (then a list of reward functions)
  Not working with constraints, but can encode constraints into its transition function
  """
  def __init__(self, S=[], A=[], T=None, r=None, alpha=None, terminal=lambda _: False, gamma=1):
    self.S = S
    self.A = A
    self.T = T
    if r is not None: self.setReward(r)
    self.alpha = alpha
    self.terminal = terminal
    self.gamma = gamma

    # can compute this to make lp more efficient
    self.transit = None
    self.invertT = None

  def setReward(self, rInput):
    if callable(rInput):
      self.r = rInput
    elif type(rInput) is list:
      self.rFuncs = map(lambda _: _[0], rInput)
      psi = map(lambda _: _[1], rInput)
      self.updatePsi(psi)
    else:
      raise Exception('unknown type of reward')

  def updatePsi(self, psi):
    self.psi = normalize(psi)
    self.r = lambda s, a: sum(rFunc(s, a) * prob for (rFunc, prob) in zip(self.rFuncs, self.psi))

  def resetInitialState(self, initS):
    """
    reset the initial state distribution to be deterministically starting from initS
    """
    self.alpha = lambda s: s == initS

  def computeTUsingTransit(self):
    """
    convert s, a -> sp to (s, a, sp) -> 1
    """
    self.T = lambda state, action, sp: 1 if sp == self.transit(state, action) else 0

  def computeInvertT(self):
    # in the flow conservation constraints in lp, we need to find out what s, a can reach sp
    # so we precompute the inverted transition function here to save time in solving lp
    self.invertT = {}

    for s in self.S:
      self.invertT[s] = []

    for s in self.S:
      if not self.terminal(s):
        for a in self.A:
          sp = self.transit(s, a)
          self.invertT[sp].append((s, a))


class DeterministicFactoredMDP(SimpleMDP):
  def __init__(self, sSets, aSets, rFunc, tFunc, s0, gamma=1, terminal=lambda s: False):
    SimpleMDP.__init__(self, A=aSets, r=rFunc, alpha=lambda s: s == s0, terminal=terminal, gamma=gamma)

    # transit(s, a) -> s'
    # the i-th component of s' is determined by tFunc[i]
    self.transit = lambda state, action: tuple([t(state, action) for t in tFunc])

    # construct the set of reachable states
    self.S = []
    buffer = [s0]
    # stop when no new states are found by one-step transitions
    while len(buffer) > 0:
      # add the last batch to S
      self.S += buffer
      newBuffer = []
      for s in buffer:
        if not terminal(s):
          for a in aSets:
            sp = self.transit(s, a)
            if not sp in self.S and not sp in newBuffer:
              newBuffer.append(sp)
      buffer = newBuffer

    # T(s, a, sp) -> prob
    self.computeTUsingTransit()
    # to make lp more efficient
    self.computeInvertT()
