import copy

import config
from algorithms.lp import lpDualGurobi, computeValue, milp
from util import computePosteriorBelief


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

    # do query iteration to locally improve this query
    if self.qi: q = self.queryIteration(q)

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
    for rewardIdx in range(len(self.mdp.psi)):
      dominatingPi = max(range(len(q)), key=lambda piIndex: self.computeValue(q[piIndex], r=self.mdp.rFuncs[rewardIdx]))
      # dominatingPi dominates rewardIdx
      dominatingIndices[dominatingPi].append(rewardIdx)

    return dominatingIndices

  def queryIteration(self, qPi):
    """
    iteratively improve one policy while fixing other policies in the query
    :param qPi:
    :return:
    """
    mdp = copy.deepcopy(self.mdp)

    oldDominatingIndices = self.findDominatedRewards(qPi)

    while True:
      newQPi = []

      for piIdx in range(self.k):
        dominatedRewards = oldDominatingIndices[piIdx]

        posteriorRewards = computePosteriorBelief(self.mdp.psi, consistentRewards=dominatedRewards)
        mdp.updatePsi(posteriorRewards)
        newPi = lpDualGurobi(mdp)['pi']
        newQPi.append(newPi)

      newDominatingIndices = self.findDominatedRewards(newQPi)
      if newDominatingIndices == oldDominatingIndices:
        # query iteration converges
        break
      else:
        oldDominatingIndices = newDominatingIndices

    return newQPi

  def findRewardSetQuery(self):
    """
    If we have only one response, find out which reward function is optimized by the first policy in qPi
    """
    qPi = self.findPolicyQuery()
    return self.findDominatedRewards(qPi)
