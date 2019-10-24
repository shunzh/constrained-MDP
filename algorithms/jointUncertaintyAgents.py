import copy
from operator import mul

import numpy
from numpy import random

from algorithms.consQueryAgents import ConsQueryAgent
from algorithms.initialSafeAgent import GreedyForSafetyAgent
from algorithms.rewardQueryAgents import GreedyConstructRewardAgent
from domains.domainConstructors import encodeConstraintIntoTransition
from util import normalize


class JointUncertaintyQueryAgent(ConsQueryAgent):
  """
  Querying under reward uncertainty and safety-constraint uncertainty
  """

  def __init__(self, mdp, consStates, goalStates=(), consProbs=None, costOfQuery=0):
    ConsQueryAgent.__init__(self, mdp, consStates, goalStates=goalStates, consProbs=consProbs)

    self.costOfQuery = costOfQuery

    # maintained after querying
    self.updatedConsProbs = copy.copy(consProbs)

  def updateFeats(self, newFreeCon=None, newLockedCon=None):
    # share some code as InitialSafeAgent, but I don't want to make this class a subclass of that
    if newFreeCon is not None:
      self.unknownCons.remove(newFreeCon)
      self.knownFreeCons.append(newFreeCon)
      self.updatedConsProbs[newFreeCon] = 1
    if newLockedCon is not None:
      self.unknownCons.remove(newLockedCon)
      self.knownLockedCons.append(newLockedCon)
      self.updatedConsProbs[newLockedCon] = 0

  def updateReward(self, possibleTrueRewardIndices):
    self.mdp.psi = self.updateARewardDistribution(possibleTrueRewardIndices, self.mdp.psi)

  def updateARewardDistribution(self, possibleTrueRewardIndices, psi):
    for rIdx in range(len(psi)):
      if rIdx not in possibleTrueRewardIndices:
        psi[rIdx] = 0
    return normalize(psi)

  def computeConsistentRewardIndices(self):
    return filter(lambda rIdx: self.mdp.psi > 0, range(len(self.mdp.psi)))


class JointUncertaintyOptimalQueryAgent(JointUncertaintyQueryAgent):
  def computeOptimalQueries(self):
    """
    find the optimal query policy by evaluating all possible query policies
    """
    pass

  def findQuery(self):
    pass


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

    agent = GreedyConstructRewardAgent(mdp, 2)
    return agent.findBinaryResponseRewardSetQuery()

  def findFeatureQuery(self):
    """
    use set-cover based algorithm and use the mean reward function (mdp.r does that)
    #fixme assume safe policies exist for now

    when safe policies exist, need to modify the original algorithm:
    computing the set structures by first removing safe dominating policies (set includeSafePolicies to True),
    that is, we want to minimize the number of queries to find *additional* dominating policies.
    """
    agent = GreedyForSafetyAgent(self.mdp, self.consStates, goalStates=self.goalCons, consProbs=self.consProbs,
                                 improveSafePis=True)
    return agent.findQuery()

  def computeEPU(self, query):
    (qType, qContent) = query
    if qType == 'F':
      feat = qContent
      return self.consProbs[feat] * self.findConstrainedOptPi(activeCons=set(self.unknownCons) - {feat,})['obj']\
           + (1 - self.consProbs[feat]) * self.findConstrainedOptPi(activeCons=self.unknownCons)['obj']
    elif qType == 'R':
      rIndices = qContent

      mdpIfTrueReward = copy.deepcopy(self.mdp)
      mdpIfTrueReward.psi = self.updateARewardDistribution(rIndices, psi=mdpIfTrueReward.psi)

      mdpIfFalseReward = copy.deepcopy(self.mdp)
      mdpIfFalseReward.psi = self.updateARewardDistribution(set(range(len(self.mdp.psi))) - set(rIndices),
                                                            psi=mdpIfFalseReward.psi)

      return sum(self.mdp.psi[_] for _ in rIndices) * self.findConstrainedOptPi(activeCons=self.unknownCons, mdp=mdpIfTrueReward)['obj'] +\
           + (1 - sum(self.mdp.psi[_] for _ in rIndices)) * self.findConstrainedOptPi(activeCons=self.unknownCons, mdp=mdpIfFalseReward)['obj']
    else:
      raise Exception('unknown query ' + query)

  def findQuery(self):
    """
    compute the myopically optimal reward query vs feature query, pose the on that has larger EPU value
    """
    rewardQuery = ('R', self.findRewardQuery())
    featureQuery = ('F', self.findFeatureQuery())

    rewardQEPU = self.computeEPU(rewardQuery)
    featureQEPU = self.computeEPU(featureQuery)

    print 'epu comparison', rewardQEPU, featureQEPU

    if rewardQEPU < self.costOfQuery and featureQEPU < self.costOfQuery:
      # stop querying
      return None
    elif rewardQEPU > featureQEPU:
      return rewardQuery
    else:
      return featureQuery


class JointUncertaintyQueryBySamplingDomPisAgent(JointUncertaintyQueryAgent):
  """
  Sample a set of dominating policies according to their probabilities of being free and their values.
  Then query the features that would make them safely-optimal.
  """
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None, costOfQuery=1):
    JointUncertaintyQueryAgent.__init__(self, mdp, consStates, goalStates=goalStates, consProbs=consProbs,
                                        costOfQuery=costOfQuery)

    # initialize objectDomPi to be None, will be computed in findQuery
    self.objectDomPi = None

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


  def sampleDomPi(self):
    """
    (re)compute all dominating policies given reward and safety uncertainty
    and then sample one
    stored in self.dompis = [(dompi, weighted_prob)]
    """
    domPisData = {}

    for rIdx in range(len(self.mdp.rewardFuncs)):
      r = self.mdp.rewardFuncs[rIdx]
      rProb = self.mdp.psi[rIdx]

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
