import copy
import random
from operator import mul

import numpy

import config
from algorithms.consQueryAgents import ConsQueryAgent
from algorithms.initialSafeAgent import GreedyForSafetyAgent
from algorithms.rewardQueryAgents import GreedyConstructRewardAgent
from util import powerset, computePosteriorBelief, printOccSA


class JointUncertaintyQueryAgent(ConsQueryAgent):
  """
  Querying under reward uncertainty and safety-constraint uncertainty
  """
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None, costOfQuery=0):
    ConsQueryAgent.__init__(self, mdp, consStates, goalStates=goalStates, consProbs=consProbs)

    self.costOfQuery = costOfQuery
    self.sizeOfRewards = len(mdp.psi)

  def updateReward(self, consistentRewards=None, inconsistentRewards=None):
    posterPsi = computePosteriorBelief(self.mdp.psi,
                                       consistentRewards=consistentRewards,
                                       inconsistentRewards=inconsistentRewards)
    self.mdp.updatePsi(posterPsi)

  def computeConsistentRewardIndices(self, psi):
    return filter(lambda rIdx: psi[rIdx] > 0, range(self.sizeOfRewards))

  def encodeConstraintIntoTransition(self, mdp):
    """
    revise the transition function in-place
    when visit a state in consStates, go to a 'sink' state with prob of pf
    """
    cons = [self.consStates[_] for _ in self.knownLockedCons + self.unknownCons]
    pfs = [0 for _ in self.knownLockedCons] + [self.consProbs[_] for _ in self.unknownCons]

    # transit is going to be set 0, so make a copy here
    transit = copy.deepcopy(mdp.transit)

    newT = {}
    for s in mdp.S:
      for a in mdp.A:
        # prob. of getting to transit(s, a)
        sp = transit(s, a)
        successProb = 1
        for (consStates, pf) in zip(cons, pfs):
          if sp in consStates:
            successProb *= pf

        newT[s, a, sp] = successProb

        # prob. of reaching sink
        newT[s, a, 'sink'] = 1 - successProb

    mdp.S.append('sink')

    mdp.T = lambda s, a, sp: newT[s, a, sp] if (s, a, sp) in newT.keys() else 0

    # make 'sink' terminal states
    terminal = copy.deepcopy(mdp.terminal)
    mdp.terminal = lambda s: s == 'sink' or terminal(s)

    # these are for deterministic transitions, they shouldn't be called (just to make sure)
    mdp.transit = None
    mdp.invertT = None

  def computeEVOI(self, query):
    """
    Compute the EVOI of the provided query (not query set)

    :param query:  can be a feature query ('F', feat) or a reward query ('R', rewards)
    """
    (qType, qContent) = query
    # if the query gives up, then epu is 0
    if qContent is None: return 0

    priorValue = self.computeCurrentSafelyOptPiValue()

    if qType == 'F':
      feat = qContent
      epu = self.consProbs[feat] * self.findConstrainedOptPi(activeCons=set(self.unknownCons) - {feat})['obj']\
          + (1 - self.consProbs[feat]) * priorValue
    elif qType == 'R':
      rIndices = qContent

      mdpIfTrueReward = copy.deepcopy(self.mdp)
      mdpIfTrueReward.updatePsi(computePosteriorBelief(mdpIfTrueReward.psi, consistentRewards=rIndices))
      posteriorValueIfTrue = self.findConstrainedOptPi(activeCons=self.unknownCons, mdp=mdpIfTrueReward)['obj']

      mdpIfFalseReward = copy.deepcopy(self.mdp)
      mdpIfFalseReward.updatePsi(computePosteriorBelief(mdpIfFalseReward.psi, inconsistentRewards=rIndices))
      posteriorValueIfFalse = self.findConstrainedOptPi(activeCons=self.unknownCons, mdp=mdpIfFalseReward)['obj']

      epu = sum(self.mdp.psi[_] for _ in rIndices) * posteriorValueIfTrue +\
          + (1 - sum(self.mdp.psi[_] for _ in rIndices)) * posteriorValueIfFalse
    else:
      raise Exception('unknown query ' + query)

    evoi = epu - priorValue
    assert evoi >= -1e-4, 'evoi value %f' % evoi
    return evoi

  def selectQueryBasedOnEVOI(self, queries, considerCost=True):
    """
    :param queries: a set of queries
    :param considerCost:  if True, then return None if the EVOI of the best query is < costOfQuery
    :return: the query that has the highest EVOI
    """
    queryAndEVOIs = []

    for query in queries:
      queryAndEVOIs.append((query, self.computeEVOI(query)))

    if config.VERBOSE: print 'select query by EVOI', queryAndEVOIs

    # break the tie randomly
    #maxEVOI = max(evoi for (q, evoi) in queryAndEVOIs)
    #optQuery = random.choice([q for (q, evoi) in queryAndEVOIs if evoi == maxEVOI])

    (optQuery, maxEVOI) = max(queryAndEVOIs, key=lambda x: x[1])

    if considerCost and maxEVOI < self.costOfQuery:
      return None
    elif optQuery[1] is None:
      return None
    else:
      return optQuery


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
    self.currentOptPiValueDict = {}

  def computeOptimalQuery(self, knownLockedCons, knownFreeCons, unknownCons, psi):
    """
    recursively compute the optimal query, return the value after query
    """
    # the key used for optQueryAndValueDict
    # use frozenset here because the order of features doesn't matter
    key = (frozenset(knownLockedCons), frozenset(knownFreeCons), frozenset(unknownCons), tuple(psi))

    if key in self.optQueryAndValueDict.keys():
      return self.optQueryAndValueDict[key]

    rewardSupports = self.computeConsistentRewardIndices(psi)
    self.imaginedMDP.updatePsi(psi)
    # compute the current safe policy
    if key in self.currentOptPiValueDict.keys():
      currentSafelyOptValue = self.currentOptPiValueDict[key]
    else:
      currentSafelyOptValue = self.findConstrainedOptPi(activeCons=list(unknownCons)+list(knownLockedCons),
                                                        addKnownLockedCons=False, mdp=self.imaginedMDP)['obj']

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

    # reward queries
    psiOfSet = lambda rSet: sum(psi[_] for _ in rSet)
    if len(rewardSupports) > 1:
      rewardQueryValues = {('R', rSet):
                           psiOfSet(rSet) * self.computeOptimalQuery(knownLockedCons, knownFreeCons, unknownCons,
                                                                     computePosteriorBelief(psi, consistentRewards=rSet))[1]
                           + (1 - psiOfSet(rSet)) * self.computeOptimalQuery(knownLockedCons, knownFreeCons, unknownCons,
                                                                             computePosteriorBelief(psi, inconsistentRewards=rSet))[1]
                           - self.costOfQuery
                           for rSet in powerset(rewardSupports, minimum=1, maximum=len(rewardSupports) - 1)}
    else:
      rewardQueryValues = {}

    queryAndValues = consQueryValues.copy()
    queryAndValues.update(rewardQueryValues)

    # also, there's an option to not pose a query
    queryAndValues[None] = currentSafelyOptValue

    optQueryAndValue = max(queryAndValues.items(), key=lambda _: _[1])

    self.optQueryAndValueDict[key] = optQueryAndValue

    return optQueryAndValue

  def findQuery(self):
    optQAndV = self.computeOptimalQuery(self.knownLockedCons, self.knownFreeCons, self.unknownCons, self.mdp.psi)

    return optQAndV[0]


class JointUncertaintyQueryByMyopicSelectionAgent(JointUncertaintyQueryAgent):
  """
  Find the myopically optimal reward query and feature query, then choose the one with higher EVOI.
  Stop when EVOI is 0.
  """
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None, costOfQuery=0):
    JointUncertaintyQueryAgent.__init__(self, mdp, consStates, goalStates, consProbs, costOfQuery)

  def findRewardQuery(self):
    """
    locally construct a rewardQueryAgent for reward query selection.
    encode consStates and pf into the transition function,
    then use greedy construction and projection to find close-to-optimal reward query
    """
    psiSupports = filter(lambda _: _ > 0, self.mdp.psi)
    # psi cannot have 0 support
    assert len(psiSupports) > 0
    # if the true reward function is known, no need to pose more reward queries
    if len(psiSupports) == 1: return None

    # going to modify the transition function in place, so make a copy of mdp
    mdp = copy.deepcopy(self.mdp)
    # encode pf into the transition probabilities
    self.encodeConstraintIntoTransition(mdp)
    rewardQueryAgent = GreedyConstructRewardAgent(mdp, 2, qi=True)

    # reward-set query has binary responses, so pose either one
    rewardQuery = rewardQueryAgent.findRewardSetQuery()[0]
    psiSupportsAndQR = sum(self.mdp.psi[idx] > 0 for idx in rewardQuery)

    if config.VERBOSE: print 'reward query', rewardQuery

    if psiSupportsAndQR == 0 or psiSupportsAndQR == len(psiSupports):
      if config.VERBOSE: print 'not reporting reward query back'
      return None
    else:
      return rewardQuery

  def findFeatureQuery(self, subsetCons=None):
    """
    locally construct an AAAI 20 agent for feature query selection.
    use set-cover based algorithm and use the mean reward function (which is mdp.r)

    when safe policies exist, need to modify the original algorithm:
    computing the set structures by first removing safe dominating policies (set includeSafePolicies to True),
    that is, we want to minimize the number of queries to find *additional* dominating policies.
    """
    featureQueryAgent = GreedyForSafetyAgent(self.mdp, self.consStates, self.goalCons, self.consProbs, improveSafePis=True)
    featureQueryAgent.knownFreeCons = self.knownFreeCons
    featureQueryAgent.knownLockedCons = self.knownLockedCons
    featureQueryAgent.unknownCons = self.unknownCons

    # recompute the set cover structure under the mean reward function (it will use self.mdp.r)
    featureQueryAgent.computePolicyRelFeats(recompute=True)
    featureQueryAgent.computeIISs(recompute=True)

    # after computing rel feats, check if it's empty. if so, nothing need to be queried.
    if len(featureQueryAgent.domPiFeats) == 0 or len(featureQueryAgent.iiss) == 0:
      if config.VERBOSE: print 'not reporting feature query back'
      return None
    else:
      query = featureQueryAgent.findQuery(subsetCons=subsetCons)
      if config.VERBOSE: print 'feature query', query
      return query

  def findQuery(self):
    """
    compute the myopically optimal reward query vs feature query, pose the on that has larger EPU value
    """
    rewardQuery = ('R', self.findRewardQuery())
    featureQuery = ('F', self.findFeatureQuery())

    return self.selectQueryBasedOnEVOI([rewardQuery, featureQuery])


class JointUncertaintyQueryBySamplingDomPisAgent(JointUncertaintyQueryAgent):
  """
  Sample a set of dominating policies according to their probabilities of being free and their values.
  Then query the features that would make them safely-optimal.
  """
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None, costOfQuery=0):
    JointUncertaintyQueryAgent.__init__(self, mdp, consStates, goalStates=goalStates, consProbs=consProbs,
                                        costOfQuery=costOfQuery)
    # initialize objectDomPiData to be None, will be computed in findQuery
    self.objectDomPiData = None

    # return as stats
    self.domPiNum = None

  class DomPiData:
    """
    For a dominating policy, we want to keep its weighted value
    (prob that it is safe, prob that the reward it optimizes is the true reward, and the value of the policy),
    the rewards it optimizes, and the constraints that it violates
    """
    def __init__(self, pi=None, weightedValue=0, optimizedRewards=[], violatedCons=[]):
      self.pi = pi
      self.weightedValue = weightedValue
      self.optimizedRewards = optimizedRewards
      self.violatedCons = violatedCons

    def __repr__(self):
      return 'DomPi score ' + str(self.weightedValue) + ' rewards optimized ' + str(self.optimizedRewards) +\
             ' rel feats ' + str(self.violatedCons)

  def findDomPi(self):
    """
    (re)compute all dominating policies given reward and safety uncertainty
    and then sample one
    stored in self.dompis = [(dompi, weighted_prob)]
    """
    domPisData = []
    allDomPis = []

    priorPi = self.computeCurrentSafelyOptPi()
    consistentRewardIndices = self.computeConsistentRewardIndices(self.mdp.psi)

    for rIndices in powerset(consistentRewardIndices, minimum=1, maximum=self.sizeOfRewards):
      rewardPositiveMDP = copy.deepcopy(self.mdp)
      rewardPositiveMDP.updatePsi(computePosteriorBelief(self.mdp.psi, consistentRewards=rIndices))

      sumOfPsi = sum(self.mdp.psi[_] for _ in rIndices)

      rewardPositiveConsAgent = ConsQueryAgent(rewardPositiveMDP, self.consStates, self.goalCons, self.consProbs,
                                               knownFreeCons=self.knownFreeCons, knownLockedCons=self.knownLockedCons)
      _, domPis = rewardPositiveConsAgent.findRelevantFeaturesAndDomPis()

      for domPi in domPis:
        relFeats = rewardPositiveConsAgent.findViolatedConstraints(domPi)

        # we are going to query about rIndices and relFeatures
        # we regard them as batch queries and compute the possible responses
        safeProb = numpy.prod([self.consProbs[feat] for feat in relFeats])
        rPositiveValue = rewardPositiveConsAgent.computeValue(domPi)

        # priorPi is feasible under relFeats since priorPi is safer (before querying)
        priorValue = rewardPositiveConsAgent.computeValue(priorPi)

        # 1 <= len(rIndices) <= sizeOfRewards
        rewardQueryNeeded = (len(rIndices) < len(consistentRewardIndices))

        # at least (relFeats) feature queries and 1 reward-set query are needed
        weightedValue = safeProb * sumOfPsi * (rPositiveValue - priorValue - self.costOfQuery * (len(relFeats) + rewardQueryNeeded))

        if domPi not in allDomPis:
          allDomPis.append(domPi)

        if weightedValue > 0:
          # only add dom pi info when it's beneficial to query about this
          domPisData.append(self.DomPiData(pi=domPi, weightedValue=weightedValue, optimizedRewards=rIndices, violatedCons=relFeats))

    if self.domPiNum is None:
      self.domPiNum = len(allDomPis)

    if len(domPisData) > 0:
      self.objectDomPiData = max(domPisData, key=lambda datum: datum.weightedValue)
    else:
      self.objectDomPiData = None

    if config.VERBOSE: print 'chosen dom pi', self.objectDomPiData

  def attemptToFindQuery(self):
    """
    try to find a query without re-sampling domPi
    return None if can't do so
    """
    queries = []

    # if not assigned, compute it
    if self.objectDomPiData is None:
      if config.VERBOSE: print 'dompi not assigned'
      return None

    # if some relevant features are now known-to-be-locked, we should find another dom pi
    if len(set(self.knownLockedCons).intersection(self.objectDomPiData.violatedCons)) > 0:
      if config.VERBOSE: print 'dompi not safe'
      return None

    consistentRewardIndices = self.computeConsistentRewardIndices(self.mdp.psi)
    # if any of the reward functions tha domPi optimizes is not possibly a true reward function
    if not set(self.objectDomPiData.optimizedRewards).issubset(consistentRewardIndices):
      if config.VERBOSE: print 'some optimized reward known to be false'
      return None

    # compute a reward query based on objective dompi
    qReward = set(self.objectDomPiData.optimizedRewards).intersection(consistentRewardIndices)
    if len(qReward) > 0 and len(qReward) < len(consistentRewardIndices):
      # if we do have something to query about, that is, not asking about all or none of the consistent rewards
      queries.append(('R', qReward))

    # compute feature queries based on objective dompi
    unknownRelFeats = set(self.objectDomPiData.violatedCons).intersection(self.unknownCons)
    queries += [('F', feat) for feat in unknownRelFeats]

    if len(queries) == 0:
      if config.VERBOSE: print 'nothing to query'
      return None
    else:
      return self.selectQueryBasedOnEVOI(queries, considerCost=False)

  def findQuery(self):
    """
    Find the dom pi with the highest score, if any exists,
    Then find its relevant feature that has the highest EVOI value.
    :return: (query type, query)
    """
    # unable to find a query, try to find a different dom pi
    self.findDomPi()
    query = self.attemptToFindQuery()

    # could be None, in which case we stop querying
    return query


class JointUncertaintyRandomQuery(JointUncertaintyQueryAgent):
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None, costOfQuery=0):
    JointUncertaintyQueryAgent.__init__(self, mdp, consStates, goalStates, consProbs, costOfQuery)

  def findQuery(self):
    featQueries = [('F', feat) for feat in self.unknownCons]

    psiSupports = filter(lambda _: _ > 0, self.mdp.psi)
    rewardQuery = ('R', filter(lambda _: random.random() > .5, psiSupports))
    noneQuery = None

    return random.choice(featQueries + [rewardQuery] + [noneQuery])
