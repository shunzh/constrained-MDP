import copy
from operator import mul

import numpy
from numpy import random

import config
from algorithms.consQueryAgents import ConsQueryAgent
from algorithms.initialSafeAgent import GreedyForSafetyAgent
from algorithms.rewardQueryAgents import GreedyConstructRewardAgent
from util import normalize, powerset


class JointUncertaintyQueryAgent(ConsQueryAgent):
  """
  Querying under reward uncertainty and safety-constraint uncertainty
  """
  def __init__(self, mdp, consStates, goalStates, consProbs, costOfQuery):
    ConsQueryAgent.__init__(self, mdp, consStates, goalStates=goalStates, consProbs=consProbs)

    self.costOfQuery = costOfQuery
    self.sizeOfRewards = len(mdp.psi)

  def updateFeats(self, newFreeCon=None, newLockedCon=None):
    # share some code as InitialSafeAgent, but I don't want to make this class a subclass of that
    if newFreeCon is not None:
      self.unknownCons.remove(newFreeCon)
      self.knownFreeCons.append(newFreeCon)
    if newLockedCon is not None:
      self.unknownCons.remove(newLockedCon)
      self.knownLockedCons.append(newLockedCon)

  def updateReward(self, consistentRewards=None, inconsistentRewards=None):
    self.mdp.updatePsi(self.updateARewardDistribution(self.mdp.psi,
                                                      consistentRewards=consistentRewards,
                                                      inconsistentRewards=inconsistentRewards))

  def updateARewardDistribution(self, psi, consistentRewards=None, inconsistentRewards=None):
    psi = copy.copy(psi)

    if inconsistentRewards is not None:
      allRewardIdx = range(self.sizeOfRewards)
      consistentRewards = set(allRewardIdx) - set(inconsistentRewards)
    assert consistentRewards is not None

    for rIdx in range(len(psi)):
      if rIdx not in consistentRewards:
        psi[rIdx] = 0
    psi = normalize(psi)
    return psi

  def computeConsistentRewardIndices(self, psi):
    return filter(lambda rIdx: psi[rIdx] > 0, range(self.sizeOfRewards))

  def computeCurrentSafelyOptPiValue(self):
    return self.findConstrainedOptPi(activeCons=self.unknownCons)['obj']


class JointUncertaintyOptimalQueryAgent(JointUncertaintyQueryAgent):
  """
  Find the optimal query policy by dynamic programming.
  Given (partition of features, possible true reward functions), it computes the immediate optimal query to pose
  """
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None, costOfQuery=0):
    JointUncertaintyQueryAgent.__init__(self, mdp, consStates, goalStates, consProbs, costOfQuery)

    # used for self.computeOptimalQuery
    self.imaginedMDP = copy.deepcopy(self.mdp)
    # for memoization
    self.optQueryAndValueDict = {}

  def computeOptimalQuery(self, knownLockedCons, knownFreeCons, unknownCons, psi):
    """
    recursively compute the optimal query, return the value after query
    """
    # the key used for optQueryAndValueDict
    # use frozenset here because the order of features doesn't matter
    key = (frozenset(knownLockedCons), frozenset(knownFreeCons), frozenset(unknownCons), tuple(psi))

    if key in self.optQueryAndValueDict.keys():
      return self.optQueryAndValueDict[key]

    # reward queries
    rewardSupports = self.computeConsistentRewardIndices(psi)
    self.imaginedMDP.updatePsi(psi)
    currentSafelyOptValue = self.findConstrainedOptPi(activeCons=unknownCons, mdp=self.imaginedMDP)['obj']

    # feature queries
    if len(unknownCons) > 0:
      consQueryValues = {('F', con):
                         self.consProbs[con] * self.computeOptimalQuery(knownLockedCons, knownFreeCons + [con],
                                                                        set(unknownCons) - {con}, psi)[1]
                         + (1 - self.consProbs[con]) * self.computeOptimalQuery(knownLockedCons + [con], knownFreeCons,
                                                                                set(unknownCons) - {con}, psi)[1]
                         - self.costOfQuery
                         for con in unknownCons}
    else:
      consQueryValues = {}

    psiOfSet = lambda rSet: sum(psi[_] for _ in rSet)
    if len(rewardSupports) > 1:
      rewardQueryValues = {('R', rSet):
                           psiOfSet(rSet) * self.computeOptimalQuery(knownLockedCons, knownFreeCons, unknownCons,
                                                                     self.updateARewardDistribution(psi, consistentRewards=rSet))[1]
                           + (1 - psiOfSet(rSet)) * self.computeOptimalQuery(knownLockedCons, knownFreeCons, unknownCons,
                                                                             self.updateARewardDistribution(psi, inconsistentRewards=rSet))[1]
                           - self.costOfQuery
                           for rSet in powerset(rewardSupports, minimum=1, maximum=len(rewardSupports) - 1)}
    else:
      rewardQueryValues = {}

    queryAndValues = consQueryValues.copy()
    queryAndValues.update(rewardQueryValues)

    # also, there's an option to no pose a query
    queryAndValues[None] = currentSafelyOptValue

    optQueryAndValue = max(queryAndValues.items(), key=lambda _: _[1])

    self.optQueryAndValueDict[key] = optQueryAndValue
    return optQueryAndValue

  def findQuery(self):
    optQAndV = self.computeOptimalQuery(self.knownLockedCons, self.knownFreeCons, self.unknownCons, self.mdp.psi)

    return optQAndV[0]


class JointUncertaintyQueryByMyopicSelectionAgent(JointUncertaintyQueryAgent):
  """
  Planning several steps into the future
  """
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None, costOfQuery=0):
    JointUncertaintyQueryAgent.__init__(self, mdp, consStates, goalStates=goalStates, consProbs=consProbs, costOfQuery=costOfQuery)

    # create two query agents specialized on reward / feature queries
    self.rewardQueryAgent = GreedyConstructRewardAgent(mdp, 2)
    self.featureQueryAgent = GreedyForSafetyAgent(mdp, consStates, goalStates, consProbs, improveSafePis=True)

  def updateFeats(self, newFreeCon=None, newLockedCon=None):
    JointUncertaintyQueryAgent.updateFeats(self, newFreeCon, newLockedCon)

    # need to update the set cover structure
    self.featureQueryAgent.updateFeats(newFreeCon, newLockedCon)
    self.featureQueryAgent.computePolicyRelFeats()
    self.featureQueryAgent.computeIISs()

  def findRewardQuery(self):
    """
    encode consStates and pf into the transition function,
    then use greedy construction and projection to find close-to-optimal reward query
    """
    psiSupports = filter(lambda _: _ > 0, self.mdp.psi)
    # psi cannot have 0 support
    assert len(psiSupports) > 0
    # if the true reward function is known, no need to pose more reward queries
    if len(psiSupports) == 1: return None

    # going to modify the transition function, so make a copy of mdp
    self.rewardQueryAgent.mdp = copy.deepcopy(self.mdp)
    self.rewardQueryAgent.mdp.encodeConstraintIntoTransition([self.consStates[_] for _ in self.knownLockedCons + self.unknownCons])

    return self.rewardQueryAgent.findBinaryResponseRewardSetQuery()

  def findFeatureQuery(self):
    """
    use set-cover based algorithm and use the mean reward function (mdp.r does that)

    when safe policies exist, need to modify the original algorithm:
    computing the set structures by first removing safe dominating policies (set includeSafePolicies to True),
    that is, we want to minimize the number of queries to find *additional* dominating policies.
    """
    # after computing rel feats, check if it's empty. if so, nothing need to be queried.
    if len(self.featureQueryAgent.domPiFeats) == 0 or len(self.featureQueryAgent.iiss) == 0: return None

    return self.featureQueryAgent.findQuery()

  def computeEVOI(self, query):
    (qType, qContent) = query
    # if the query gives up, then epu is 0
    if qContent is None: return 0

    priorValue = self.computeCurrentSafelyOptPiValue()

    if qType == 'F':
      feat = qContent
      epu = self.consProbs[feat] * self.findConstrainedOptPi(activeCons=set(self.unknownCons) - {feat})['obj']\
          + (1 - self.consProbs[feat]) * self.findConstrainedOptPi(activeCons=self.unknownCons)['obj']
    elif qType == 'R':
      rIndices = qContent

      mdpIfTrueReward = copy.deepcopy(self.mdp)
      mdpIfTrueReward.updatePsi(self.updateARewardDistribution(mdpIfTrueReward.psi, consistentRewards=rIndices))
      posteriorValueIfTrue = self.findConstrainedOptPi(activeCons=self.unknownCons, mdp=mdpIfTrueReward)['obj']

      mdpIfFalseReward = copy.deepcopy(self.mdp)
      mdpIfFalseReward.updatePsi(self.updateARewardDistribution(mdpIfFalseReward.psi, inconsistentRewards=rIndices))
      posteriorValueIfFalse = self.findConstrainedOptPi(activeCons=self.unknownCons, mdp=mdpIfFalseReward)['obj']

      epu = sum(self.mdp.psi[_] for _ in rIndices) * posteriorValueIfTrue +\
          + (1 - sum(self.mdp.psi[_] for _ in rIndices)) * posteriorValueIfFalse
    else:
      raise Exception('unknown query ' + query)

    if config.VERBOSE: print epu, '-', priorValue
    evoi = epu - priorValue
    assert evoi >= 0
    return evoi

  def findQuery(self):
    """
    compute the myopically optimal reward query vs feature query, pose the on that has larger EPU value
    """
    rewardQuery = ('R', self.findRewardQuery())
    featureQuery = ('F', self.findFeatureQuery())

    rewardQEVOI = self.computeEVOI(rewardQuery)
    featureQEVOI = self.computeEVOI(featureQuery)

    if config.VERBOSE:
      print 'EVOI', rewardQuery, rewardQEVOI
      print 'EVOI', featureQuery, featureQEVOI

    if rewardQEVOI <= self.costOfQuery and featureQEVOI <= self.costOfQuery:
      # stop querying
      return None
    elif rewardQEVOI > featureQEVOI:
      return rewardQuery
    else:
      return featureQuery


class JointUncertaintyQueryBySamplingDomPisAgent(JointUncertaintyQueryAgent):
  """
  Sample a set of dominating policies according to their probabilities of being free and their values.
  Then query the features that would make them safely-optimal.
  """
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None, costOfQuery=0, heuristicID=0):
    JointUncertaintyQueryAgent.__init__(self, mdp, consStates, goalStates=goalStates, consProbs=consProbs,
                                        costOfQuery=costOfQuery)

    # initialize objectDomPi to be None, will be computed in findQuery
    self.objectDomPi = None

    self.heuristicID = heuristicID

  class DomPiData:
    """
    For a dominating policy, we want to keep its weighted value
    (prob that it is safe, prob that the reward it optimizes is the true reward, and the value of the policy),
    the rewards it optimizes, and the constraints that it violates
    """
    def __init__(self):
      self.pi = None
      self.weightedValue = 0
      self.optimizedRewards = []
      self.violatedCons = None

    def __repr__(self):
      return 'DomPi score ' + str(self.weightedValue) + ' rewards optimized ' + str(self.optimizedRewards) +\
             ' rel feats ' + str(self.violatedCons)

  def sampleDomPi(self):
    """
    (re)compute all dominating policies given reward and safety uncertainty
    and then sample one
    stored in self.dompis = [(dompi, weighted_prob)]
    """
    domPisData = []

    priorValue = self.computeCurrentSafelyOptPiValue()
    consistentRewardIndices = self.computeConsistentRewardIndices(self.mdp.psi)

    for rIndices in powerset(consistentRewardIndices, minimum=1, maximum=self.sizeOfRewards - 1):
      rewardCertainMDP = copy.deepcopy(self.mdp)
      rewardCertainMDP.updatePsi(rIndices)
      sumOfPsi = sum(self.mdp.psi[_] for _ in rIndices)

      rewardCertainConsAgent = ConsQueryAgent(rewardCertainMDP, self.consStates, self.goalCons, self.consProbs,
                                              knownFreeCons=self.knownFreeCons, knownLockedCons=self.knownLockedCons)
      _, domPis = rewardCertainConsAgent.findRelevantFeaturesAndDomPis()

      for domPi in domPis:
        relFeats = rewardCertainConsAgent.findViolatedConstraints(domPi)

        domPisDatum = self.DomPiData()
        domPisDatum.pi = domPi
        domPisDatum.optimizedRewards = rIndices
        domPisDatum.violatedCons = relFeats

        if self.heuristicID == 0:
          piValue = rewardCertainConsAgent.computeValue(domPi)
          safeProb = reduce(mul, [self.consProbs[feat] for feat in relFeats], 1)

          # at least (relFeats) feature queries and (1) reward-set query are needed
          domPisDatum.weightedValue += max(safeProb * sumOfPsi * (piValue - priorValue - (len(relFeats) + 1) * self.costOfQuery),
                                           0)
        elif self.heuristicID == 1:
          domPisDatum.weightedValue = 1.0
        else:
          raise Exception('unknown heuristicID ' + str(self.heuristicID))

    if self.heuristicID == 0:
      for datum in domPisData:
        datum[1].weightedValue = max(datum.weightedValue - priorValue, 0)

    # normalize values
    sumOfAllValues = sum([data[1].weightedValue for data in domPisData])

    # FIXME are these termination conditions complete?
    if len(domPisData) == 0 or sumOfAllValues == 0:
      self.objectDomPi = None
      return

    for datum in domPisData:
      datum[1].weightedValue /= sumOfAllValues

    self.objectDomPiData = numpy.random.choice(domPisData, p=[data[1].weightedValue for data in domPisData])
    if config.VERBOSE: print 'chosen dom pi', self.objectDomPiData

  def objectDomPiIsConsistent(self):
    """
    If the reward functions the current objectDomPi optimize are ruled out, or the current objectDomPi is knwon to be unsafe,
    then re-compute the set of dominating policies
    """
    # if the response is inconsistent with self.objectDomPi,
    # we void the current object dom pi, findQuery will recompute the object dom pi
    consistentRewardIndices = self.computeConsistentRewardIndices(self.mdp.psi)

    return len(set(self.knownLockedCons).intersection(self.objectDomPiData.violatedCons)) == 0 \
      and len(set(consistentRewardIndices).intersection(self.objectDomPiData.optimizedRewards)) > 0

  def findQuery(self):
    """
    sample some dominating policies, find the most useful query?
    :return: (query type, query)
    """
    # sample dom pis, find what can make them be the safely optimal one
    self.sampleDomPi()

    if self.objectDomPi is None:
      return None # safe policies not exist

    relFeats = self.objectDomPiData.violatedCons
    unknownRelFeats = set(relFeats).intersection(self.unknownCons)
    # fixme better ways to choose queries and decide when to stop?
    if len(unknownRelFeats) > 0:
      # pose constraint queries if any relevant features are unknown
      return ('F', random.choice(list(unknownRelFeats)))
    else:
      # pose reward queries aiming to show that the rewards it optimizes is correct
      consistentRewardIndices = self.computeConsistentRewardIndices(self.mdp.psi)
      assert len(consistentRewardIndices) > 0

      # if true, we know the robot's optimzies the true safely-optimal policy
      if set(consistentRewardIndices).issubset(self.objectDomPiData.optimizedRewards): return None

      # no reward queries needed if no reward uncertainty
      qReward = set(self.objectDomPiData.optimizedRewards).intersection(consistentRewardIndices)

      return ('R', list(qReward))
