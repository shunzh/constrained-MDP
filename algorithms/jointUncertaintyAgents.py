import copy
from operator import mul

import numpy
from numpy import random

from algorithms.consQueryAgents import ConsQueryAgent


class JointUncertaintyQueryAgent(ConsQueryAgent):
  """
  Querying under reward uncertainty and safety-constraint uncertainty
  """
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None):
    ConsQueryAgent.__init__(self, mdp, consStates, goalStates=goalStates, consProbs=consProbs)

  def updateFeats(self, newFreeCon=None, newLockedCon=None):
    #FIXME share some code as InitialSafeAgent, but I don't want to make this class a subclass of that
    if newFreeCon is not None:
      self.unknownCons.remove(newFreeCon)
      self.knownFreeCons.append(newFreeCon)
    if newLockedCon is not None:
      self.unknownCons.remove(newLockedCon)
      self.knownLockedCons.append(newLockedCon)

  def updateReward(self, possibleTrueRewardsIndices):
    sumOfProbs = 0
    for rIdx in range(len(self.mdp.rSetAndProb)):
      if rIdx not in possibleTrueRewardsIndices:
        self.mdp.rSetAndProb[rIdx] = (self.mdp.rSetAndProb[rIdx][0], 0)
      sumOfProbs += self.mdp.rSetAndProb[rIdx][1]

    # you can't get 0 prob mass
    assert sumOfProbs > 0

    # normalize reward probs
    for rIdx in  range(len(self.mdp.rSetAndProb)):
      self.mdp.rSetAndProb[rIdx] = (self.mdp.rSetAndProb[rIdx][0], self.mdp.rSetAndProb[rIdx][1] / sumOfProbs)


class JointUncertaintyQueryByMyopicSelectionAgent(JointUncertaintyQueryAgent):
  """
  Planning several steps into the future
  """
  def findQuery(self):
    pass


class DomPiData:
  """
  Used by the querying agent that samples dominating policies
  For a dominating policy, we want to keep its weighted value
  (prob that it is safe, prob that the reward it optimizes is the true reward, and the value of the policy),
  the rewards it optimizes, and the constraints that it violates
  """
  def __init__(self):
    self.weightedValue = 0
    self.optimizedRewards = []
    self.violatedCons = None

class JointUncertaintyQueryBySamplingDomPisAgent(JointUncertaintyQueryAgent):
  """
  Sample a set of dominating policies according to their probabilities of being free and their values.
  Then query the features that would make them safely-optimal.
  """
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None):
    JointUncertaintyQueryAgent.__init__(self, mdp, consStates, goalStates, consProbs)

    # pre-compute all dominating policies
    self.findDomPis()

    # aim to show that objectReardFunc is the true reward function, and objectDomPi is the safely-optimal policy
    self.objectDomPi = None

  def updateFeats(self, newFreeCon=None, newLockedCon=None):
    JointUncertaintyQueryAgent.updateFeats(self, newFreeCon=newFreeCon, newLockedCon=newLockedCon)

    # if the response is inconsistent with self.objectDomPi,
    # we void the current object dom pi, findQuery will recompute the object dom pi
    rSet = map(lambda _: _[0], self.mdp.rSetAndProb)
    if len(set(self.knownLockedCons).intersection(self.domPisData[self.objectDomPi].violatedCons)) > 0\
      or len(set(rSet).intersection(self.domPisData[self.objectDomPi].optimizedRewards)) == 0:
      self.objectDomPi = None

  def findDomPis(self):
    """
    find all dominating policies given reward and safety uncertainty
    stored in self.dompis = [(dompi, weighted_prob)]
    """
    self.domPisData = {}

    for rIdx in range(len(self.mdp.rSetAndProb)):
      (r, rProb) = self.mdp.rSetAndProb[rIdx]

      rewardCertainMDP = copy.deepcopy(self.mdp)
      rewardCertainMDP.setReward(r)

      rewardCertainConsAgent = ConsQueryAgent(rewardCertainMDP, self.consStates, goalStates=self.goalCons, consProbs=self.consProbs)
      _, domPis = rewardCertainConsAgent.findRelevantFeaturesAndDomPis()

      for domPi in domPis:
        piValue = rewardCertainConsAgent.computeValue(domPi)
        relFeats = rewardCertainConsAgent.findViolatedConstraints(domPi)
        safeProb = reduce(mul, [self.consProbs[feat] for feat in relFeats], 1)

        if domPi not in self.domPisData.keys():
          self.domPisData[domPi] = DomPiData()
          self.domPisData[domPi].violatedCons = relFeats

        self.domPisData[domPi].weightedValue += safeProb * rProb * piValue
        self.domPisData[domPi].optimizedRewards.append(rIdx)

    # normalize values
    sumOfAllValues = sum([data.weightedValue for data in self.domPisData.values()])
    for domPi in self.domPisData.keys():
      self.domPisData[domPi].weightedValue /= sumOfAllValues

  def sampleDomPi(self):
    return numpy.random.choice(self.domPisData.keys(), p=[data.weightedValue for data in self.domPisData.values()])

  def findQuery(self):
    """
    sample some dominating policies, find the most useful query?
    :return: (query type, query)
    """
    # sample dom pis, find what can make them be the safely optimal one
    if self.objectDomPi is None:
      self.objectDomPi = self.sampleDomPi()

    relFeats = self.domPisData[self.objectDomPi].violatedCons
    unknownRelFeats =  set(relFeats).intersection(self.unknownCons)
    if len(unknownRelFeats) > 0:
      # pose constraint queries if any relevant features are unknown
      return ('F', random.choice(unknownRelFeats))
    else:
      # pose reward queries aiming to show that the rewards it optimize is correct
      rSet = map(lambda _: _[0], self.mdp.rSetAndProb)
      return ('R', set(self.domPisData[self.objectDomPi].optimizedRewards).intersection(rSet))
