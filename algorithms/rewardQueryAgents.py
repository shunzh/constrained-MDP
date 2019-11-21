import copy

import config
from algorithms.lp import lpDualGurobi, computeValue, milp
from util import computePosteriorBelief, printOccSA


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
    meanRewardOptPi = lpDualGurobi(self.mdp)['pi']
    q = [meanRewardOptPi]

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

  def findRewardSetQuery(self, qPi=None):
    """
    :param qPi: provided if already computed
    :return: [indices of rewards being dominateded by policy, for policy in qPi]
    """
    if qPi is None: qPi = self.findPolicyQuery()

    dominatingIndices = [[] for _ in qPi]
    for rewardIdx in range(len(self.mdp.psi)):
      dominatingPi = max(range(len(qPi)), key=lambda piIndex: self.computeValue(qPi[piIndex], r=self.mdp.rFuncs[rewardIdx]))
      # dominatingPi dominates rewardIdx
      dominatingIndices[dominatingPi].append(rewardIdx)

    return dominatingIndices

  def computeEUS(self, qPi, qR):
    ret = 0
    for (pi, rs) in zip(qPi, qR):
      for rIdx in rs:
        piValue = self.computeValue(pi, self.mdp.rFuncs[rIdx])
        if config.VERBOSE: print 'reward', rIdx, 'value', piValue

        ret += self.mdp.psi[rIdx] * piValue

    return ret

  def queryIteration(self, qPi):
    """
    iteratively improve one policy while fixing other policies in the query
    :return: local optimum policy query
    """
    mdp = copy.deepcopy(self.mdp)

    oldDominatingIndices = self.findRewardSetQuery(qPi)
    oldEUS = self.computeEUS(qPi, oldDominatingIndices)

    while True:
      newQPi = []

      for piIdx in range(self.k):
        dominatedRewards = oldDominatingIndices[piIdx]

        posteriorRewards = computePosteriorBelief(self.mdp.psi, consistentRewards=dominatedRewards)
        mdp.updatePsi(posteriorRewards)
        newPi = lpDualGurobi(mdp)['pi']
        newQPi.append(newPi)

      newDominatingIndices = self.findRewardSetQuery(newQPi)

      newEUS = self.computeEUS(newQPi, newDominatingIndices)

      if newEUS <= oldEUS + 1e-3:
        # query iteration converges
        break
      else:
        oldDominatingIndices = newDominatingIndices
        oldEUS = newEUS

    return newQPi

