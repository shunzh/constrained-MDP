
class SimpleMDP:
  """
  An MDP object. All fields are initialized in __init__.
  """
  def __init__(self, S=[], A=[], T=None, r=None, alpha=None, terminal=lambda _: False, gamma=1):
    """
    r is set if reward function is known,
    """
    self.S = S
    self.A = A
    self.T = T
    self.alpha = alpha
    self.terminal = terminal
    self.gamma = gamma

    if r is not None: self.setReward(r)

    # (s, a) -> s'. convenient for deterministic transition functions
    self.transit = None
    # the invert transition function. don't compute this by default
    self.invertT = None

  def setReward(self, r):
    if callable(r):
      self.r = r

      self.rewardFuncs = [r,]
      self.psi = [1,]
    elif type(r) is list:
      self.rewardFuncs = map(lambda _: _[0], r)
      self.psi = map(lambda _: _[1], r)

      self.r = lambda s, a: sum(rFunc(s, a) * prob for (rFunc, prob) in zip(self.rewardFuncs, self.psi))
    else:
      raise Exception('unknown type of r ' + str(type(r)))

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
  mdp.setReward(rFunc)

  # transit(s, a) -> s'
  # the i-th component of s' is determined by tFunc[i]
  transit = lambda state, action: tuple([t(state, action) for t in tFunc])
  # transit(s, a) -> sp
  mdp.transit = transit
  # T(s, a, sp) -> prob
  mdp.T = lambda state, action, sp: 1 if sp == transit(state, action) else 0

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

  # in the flow conservation constraints in lp, we need to find out what s, a can reach sp
  # so we precompute the inverted transition function here to save time in solving lp
  mdp.invertT = {}
  for s in mdp.S:
    mdp.invertT[s] = []
  for s in mdp.S:
    if not terminal(s):
      for a in mdp.A:
        sp = transit(s, a)
        mdp.invertT[sp].append((s, a))

  return mdp


def encodeConstraintIntoTransition(mdp, cons, pfs):
  """
  Hack: ignore pf for now

  Add a sink state with 0 reward.
  for each feature, with prob. 1 - pf(\phi), the transition goes to the sink state (meaning the current action is not feasible)
  here assume rewards are non-negative: so going to the sink with reward 0 is bad.

  :param mdp: a SimpleMDP objective
  :param cons: lists of sets of states that should not be visited
  :return: None, mdp.T is changed in place
  """
  forbiddenStates = []
  for s in mdp.S:
    if any(s in consStates for consStates in cons):
      forbiddenStates.append(s)

  def newTransFunc(s, a, sp):
    if sp == mdp.transit(s, a) and s not in forbiddenStates: return 1
    elif sp == s and s in forbiddenStates: return 0
    else: return 0

  mdp.T = newTransFunc

  # inverted transition functions not computed
  mdp.invertT = None
