import copy
from operator import mul

import numpy
from numpy import random

from algorithms.consQueryAgents import ConsQueryAgent
from algorithms.initialSafeAgent import GreedyForSafetyAgent
from algorithms.rewardQueryAgents import MILPAgent
from domains.domainConstructors import encodeConstraintIntoTransition


class JointUncertaintyQueryAgent(ConsQueryAgent):
  """
  Querying under reward uncertainty and safety-constraint uncertainty
  """

  def __init__(self, mdp, consStates, goalStates=(), consProbs=None):
    ConsQueryAgent.__init__(self, mdp, consStates, goalStates=goalStates, consProbs=consProbs)

  def updateFeats(self, newFreeCon=None, newLockedCon=None):
    # share some code as InitialSafeAgent, but I don't want to make this class a subclass of that
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
    for rIdx in range(len(self.mdp.rSetAndProb)):
      self.mdp.rSetAndProb[rIdx] = (self.mdp.rSetAndProb[rIdx][0], self.mdp.rSetAndProb[rIdx][1] / sumOfProbs)

  def computeConsistentRewardIndices(self):
    return filter(lambda rIdx: self.mdp.rSetAndProb[rIdx][1] > 0, range(len(self.mdp.rSetAndProb)))


class JointUncertaintyQueryByMyopicSelectionAgent(JointUncertaintyQueryAgent):
  """
  Planning several steps into the future
  """
  def findRewardQuery(self):
    """
    encode consStates and pf into the transition function,
    then use greedy construction and projection to find close-to-optimal reward query
    """
    mdp = copy.deepcopy(self.mdp)
    encodeConstraintIntoTransition(mdp, self.consStates, self.consProbs)

    agent = MILPAgent(mdp, 2)
    agent.learn()

  def findFeatureQuery(self):
    """
    use set-cover based algorithm and use the mean reward function (mdp.r does that)

    when safe policies exist, need to modify the original algorithm:
    computing the set structures by first removing safe dominating policies (set includeSafePolicies to True),
    that is, we want to minimize the number of queries to find *additional* dominating policies.
    """
    agent = GreedyForSafetyAgent(self.mdp, self.consStates, goalStates=self.goalCons, consProbs=self.consProbs,
                                 includeSafePolicies=True)
    return agent.findQuery()

  def findQuery(self):
    """
    compute the myopically optimal reward query vs feature query, pose the on that has larger EPU value
    """
    rewardQuery = self.findRewardQuery()
    featureQuery = self.findFeatureQuery()


class JointUncertaintyQueryBySamplingDomPisAgent(JointUncertaintyQueryAgent):
  """
  Sample a set of dominating policies according to their probabilities of being free and their values.
  Then query the features that would make them safely-optimal.
  """
  class DomPiData:
    """
    For a dominating policy, we want to keep its weighted value
    (prob that it is safe, prob that the reward it optimizes is the true reward, and the value of the policy),
    the rewards it optimizes, and the constraints that it violates
    """
    def __init__(self):
      self.weightedValue = 0
      self.optimizedRewards = []
      self.violatedCons = None

  def __init__(self, mdp, consStates, goalStates=(), consProbs=None):
    JointUncertaintyQueryAgent.__init__(self, mdp, consStates, goalStates, consProbs)

    # initialize objectDomPi to be None, will be computed in findQuery
    self.objectDomPi = None

  def sampleDomPi(self):
    """
    (re)compute all dominating policies given reward and safety uncertainty
    and then sample one
    stored in self.dompis = [(dompi, weighted_prob)]
    """
    domPisData = {}

    for rIdx in range(len(self.mdp.rSetAndProb)):
      (r, rProb) = self.mdp.rSetAndProb[rIdx]

      rewardCertainMDP = copy.deepcopy(self.mdp)
      rewardCertainMDP.setReward(r)

      rewardCertainConsAgent = ConsQueryAgent(rewardCertainMDP, self.consStates, goalStates=self.goalCons,
                                              consProbs=self.consProbs)
      _, domPis = rewardCertainConsAgent.findRelevantFeaturesAndDomPis()

      for domPi in domPis:
        piValue = rewardCertainConsAgent.computeValue(domPi)
        relFeats = rewardCertainConsAgent.findViolatedConstraints(domPi)
        safeProb = reduce(mul, [self.consProbs[feat] for feat in relFeats], 1)

        domPiHashable = frozenset(domPi.items())
        if domPiHashable not in domPisData.keys():
          domPisData[domPiHashable] = self.DomPiData()
          domPisData[domPiHashable].violatedCons = relFeats

        domPisData[domPiHashable].weightedValue += safeProb * rProb * piValue
        domPisData[domPiHashable].optimizedRewards.append(rIdx)

    # normalize values
    sumOfAllValues = sum([data.weightedValue for data in domPisData.values()])
    for domPiHashable in domPisData.keys():
      domPisData[domPiHashable].weightedValue /= sumOfAllValues

    self.objectDomPi = numpy.random.choice(domPisData.keys(), p=[data.weightedValue for data in domPisData.values()])
    self.objectDomPiData = copy.copy(domPisData[self.objectDomPi]) # hopefully python will then free domPisData

  def objectDomPiIsConsistent(self):
    """
    If the reward functions the current objectDomPi optimize are ruled out, or the current objectDomPi is knwon to be unsafe,
    then re-compute the set of dominating policies
    """
    # if the response is inconsistent with self.objectDomPi,
    # we void the current object dom pi, findQuery will recompute the object dom pi
    consistentRewardIndices = self.computeConsistentRewardIndices()
    #print 'known locked cons', self.knownLockedCons
    #print 'consistent reward indices', consistentRewardIndices
    return len(set(self.knownLockedCons).intersection(self.objectDomPiData.violatedCons)) == 0 \
      and len(set(consistentRewardIndices).intersection(self.objectDomPiData.optimizedRewards)) > 0

  def findQuery(self):
    """
    sample some dominating policies, find the most useful query?
    :return: (query type, query)
    """
    # sample dom pis, find what can make them be the safely optimal one
    if self.objectDomPi is None or not self.objectDomPiIsConsistent():
      self.sampleDomPi()

    relFeats = self.objectDomPiData.violatedCons
    unknownRelFeats = set(relFeats).intersection(self.unknownCons)
    if len(unknownRelFeats) > 0:
      # pose constraint queries if any relevant features are unknown
      return ('F', random.choice(list(unknownRelFeats)))
    else:
      # pose reward queries aiming to show that the rewards it optimize is correct
      consistentRewardIndices = self.computeConsistentRewardIndices()
      return ('R', set(self.objectDomPiData.optimizedRewards).intersection(consistentRewardIndices))
