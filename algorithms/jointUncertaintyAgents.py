import copy
from operator import mul

from numpy import random

from algorithms.consQueryAgents import ConsQueryAgent


class JointUncertaintyQueryAgent:
  """
  Querying under reward uncertainty and safety-constraint uncertainty
  """
  def __init__(self, mdp, consStates, consProbs):
    self.mdp = mdp
    self.consStates = consStates
    self.consProbs = consProbs

  def updateCons(self, newFreeCon=None, newLockedCon=None):
    pass

  def updateReward(self, possibleTrueRewards):
    self.mdp.setReward(possibleTrueRewards)


class JointUncertaintyQueryByMyopicPlanningAgent(JointUncertaintyQueryAgent):
  """
  Planning several steps into the future
  """
  def findQuery(self):
    pass


class JointUncertaintyQueryBySamplingDomPisAgent(JointUncertaintyQueryAgent):
  """
  Sample a set of dominating policies according to their probabilities of being free and their values.
  Then query the features that would make them safely-optimal.
  """
  def __init__(self, mdp, consStates, consProbs, numOfSampledPis=10):
    JointUncertaintyQueryAgent.__init__(self, mdp, consStates, consProbs)

    self.numOfSampledPis = numOfSampledPis

  def findDomPi(self):
    """
    find all dominating policies given reward and safety uncertainty
    stored in self.dompis = [(dompi, weighted_prob)]
    """
    self.domPisAndWeightedValue = {}

    for r in self.mdp.rSet:
      rewardCertainMDP = copy.deepcopy(self.mdp)
      rewardCertainMDP.setReward(r)

      rewardCertainConsAgent = ConsQueryAgent(rewardCertainMDP, self.consStates)
      _, domPis = rewardCertainConsAgent.findRelevantFeaturesAndDomPis()

      for domPi in domPis:
        piValue = rewardCertainConsAgent.computeValue(domPi)
        relFeats = rewardCertainConsAgent.findViolatedConstraints(domPi)
        safeProb = reduce(mul, [self.consProbs[feat] for feat in relFeats], 1)

        if domPi not in self.domPisAndWeightedValue.keys():
          self.domPisAndWeightedValue[domPi] = 0

        self.domPisAndWeightedValue[domPi] += safeProb * piValue

    # normalize values
    sumOfAllValues = sum(self.domPisAndWeightedValue.values())
    for domPi in self.domPisAndWeightedValue.keys():
      self.domPisAndWeightedValue[domPi] /= sumOfAllValues

  def sampleDomPi(self):
    return random.choice(self.domPisAndWeightedValue.keys(), self.domPisAndWeightedValue.values())

  def findQuery(self):
    """
    sample some dominating policies, find the most useful query?
    :return: (query type, query)
    """
    # sample dom pis, find what can make them be the safely optimal one
    sampledDomPis = [self.sampleDomPi() for _ in range(self.numOfSampledPis)]

