import copy

import config
from algorithms.consQueryAgents import ConsQueryAgent
from algorithms.lp import lpDualGurobi, computeValue, milp, jointUncertaintyMilp
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
    q = [self.findOptPolicyUnderMeanRewards()]

    # start adding following policies
    for i in range(1, self.k):
      x = self.findNextPolicy(q)
      q.append(x)

    # do query iteration to locally improve this query
    if self.qi: q = self.queryIteration(q)

    return q

  def findOptPolicyUnderMeanRewards(self, psi=None):
    if psi is not None:
      mdp = copy.deepcopy(self.mdp)
      mdp.updatePsi(psi)
    else:
      # use the current psi
      mdp = self.mdp

    return lpDualGurobi(mdp)['pi']

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
      qPiValues = {piIdx: GreedyConstructRewardAgent.computeValue(self, qPi[piIdx], r=self.mdp.rFuncs[rewardIdx]) for piIdx in range(len(qPi))}
      if config.VERBOSE: print 'r', rewardIdx, 'pi values', qPiValues
      dominatingPi = max(qPiValues.keys(), key=lambda piIdx: qPiValues[piIdx])
      dominatingIndices[dominatingPi].append(rewardIdx)

    return dominatingIndices

  def computeEUS(self, qPi, qR=None):
    """
    :param qR: provided if pre-computed dominated rewards
    :return:
    """
    if qR is None: qR = self.findRewardSetQuery(qPi)

    ret = 0
    for (pi, rs) in zip(qPi, qR):
      for rIdx in rs:
        piValue = self.computeValue(pi, self.mdp.rFuncs[rIdx])

        ret += self.mdp.psi[rIdx] * piValue

    return ret

  def queryIteration(self, qPi):
    """
    iteratively improve one policy while fixing other policies in the query
    :return: local optimum policy query
    """
    #map(lambda _: printOccSA(_), qPi)
    oldDominatingIndices = self.findRewardSetQuery(qPi)
    oldEUS = self.computeEUS(qPi, oldDominatingIndices)

    while True:
      newQPi = []

      for piIdx in range(len(qPi)):
        dominatedRewards = oldDominatingIndices[piIdx]

        if len(dominatedRewards) > 0:
          posteriorBelief = computePosteriorBelief(self.mdp.psi, consistentRewards=dominatedRewards)
          newPi = self.findOptPolicyUnderMeanRewards(posteriorBelief)
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


