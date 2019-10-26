import config
from algorithms.lp import lpDualGurobi, computeValue, milp


class GreedyConstructRewardAgent:
  def __init__(self, mdp, k):
    """
    qi: query iteration
    """
    self.mdp = mdp
    self.k = k

  def computeValue(self, x, r=None):
    """
    compute the value of policy x. it computes the dot product between x and r
    """
    if r is None: r = self.mdp.r
    return computeValue(x, r, self.mdp.S, self.mdp.A)

  def findPolicyQuery(self):
    # start with the prior optimal policy
    q = [lpDualGurobi(self.mdp)['pi']]

    # start adding following policies
    for i in range(1, self.k):
      if config.VERBOSE: print 'iter.', i
      x = self.findNextPolicy(q)
      q.append(x)

    # if asking policies directly, then return q
    # return q, objValue # THIS RETURNS EUS, NOT EPU
    return q

  def findNextPolicy(self, q):
    maxV = []
    rewardCandNum = len(self.mdp.psi)
    for rewardIdx in xrange(rewardCandNum):
      maxV.append(max([self.computeValue(pi, r=self.mdp.rFuncs[rewardIdx]) for pi in q]))

    # solve a MILP problem
    return milp(self.mdp, maxV)

  def findBinaryResponseRewardSetQuery(self):
    """
    If we have only one response, find out which reward function is optimized by the first policy in qPi
    """
    qPi = self.findPolicyQuery()

    rewardIndicesSet = []
    for rewardIdx in range(len(self.mdp.psi)):
      qPiValues = [self.computeValue(qPi[0], r=self.mdp.rFuncs[rewardIdx]),
                   self.computeValue(qPi[1], r=self.mdp.rFuncs[rewardIdx])]
      print rewardIdx, qPiValues
      if qPiValues[0] > qPiValues[1]:
        rewardIndicesSet.append(rewardIdx)

    return rewardIndicesSet
