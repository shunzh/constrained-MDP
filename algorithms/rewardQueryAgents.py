import config
from algorithms.lp import lpDualGurobi, computeValue, milp


class GreedyConstructRewardAgent:
  def __init__(self, mdp, k, qi=False):
    """
    :param k: number of queries
    :param qi: query iteration
    """
    self.mdp = mdp
    self.k = k
    self.qi = qi

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

  def findDominatedRewards(self, q):
    """
    :return: [indices of rewards being dominateded, for policy in q]
    """
    dominatingIndices = [[] for _ in q]
    for rewardIdx in range(self.mdp.psi):
      dominatingPi = max(range(len(q)), key=lambda piIndex: self.computeValue(q[piIndex], r=self.mdp.rFuncs(rewardIdx)))
      dominatingIndices[dominatingPi].append(rewardIdx)

    return dominatingIndices

  def queryIteration(self, qPi):
    """
    iteratively improve one policy while fixing other policies in the query
    :param qPi:
    :return:
    """
    oldDominatingIndices = self.findDominatedRewards(qPi)

    for piIdx in range(self.k):


  def findBinaryResponseRewardSetQuery(self):
    """
    If we have only one response, find out which reward function is optimized by the first policy in qPi
    """
    qPi = self.findPolicyQuery()

    dominatingIndices = self.findDominatedRewards(qPi)

    return dominatingIndices[0]
