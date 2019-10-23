import config
from algorithms.lp import lpDualGurobi, computeValue, milp


class GreedyConstructRewardAgent:
  def __init__(self, mdp, k):
    """
    qi: query iteration
    """
    self.mdp = mdp
    self.k = k

  def computeValue(self, x):
    """
    compute the value of policy x. it computes the dot product between x and r
    """
    return computeValue(x, self.mdp.r, self.mdp.S, self.mdp.A)

  def findQuery(self):
    # start with the prior optimal policy
    q = [lpDualGurobi(self.mdp)]
    objValue = None  # k won't be 1, fine

    # start adding following policies
    for i in range(1, self.k):
      if config.VERBOSE: print 'iter.', i
      x = self.findNextPolicy(self.mdp, q)
      q.append(x)

    # if asking policies directly, then return q
    # return q, objValue # THIS RETURNS EUS, NOT EPU
    return q, objValue

  def findNextPolicy(self, mdp, q):
    maxV = []
    for rewardId in xrange(rewardCandNum):
      maxV.append(max([self.computeValue(pi) for pi in q]))

    # solve a MILP problem
    return milp(S, A, R, T, s0, psi, maxV)

