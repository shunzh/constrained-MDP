import copy
import random
from operator import mul

import config
from algorithms.consQueryAgents import ConsQueryAgent
from algorithms.initialSafeAgent import GreedyForSafetyAgent
from algorithms.rewardQueryAgents import GreedyConstructRewardAgent, GreedyConstructRewardWithConsCostAgent
from util import powerset, computePosteriorBelief, printOccSA


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
    posterPsi = computePosteriorBelief(self.mdp.psi,
                                       consistentRewards=consistentRewards,
                                       inconsistentRewards=inconsistentRewards)
    self.mdp.updatePsi(posterPsi)

  def computeConsistentRewardIndices(self, psi):
    return filter(lambda rIdx: psi[rIdx] > 0, range(self.sizeOfRewards))

  def computeCurrentSafelyOptPi(self):
    return self.findConstrainedOptPi(activeCons=self.unknownCons)['pi']

  def computeCurrentSafelyOptPiValue(self):
    return self.findConstrainedOptPi(activeCons=self.unknownCons)['obj']

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

  def addFeatQueryCostToReward(self, mdp):
    """
    discourage the robot from violating safety constraints but put the cost of query into the reward.
    looks like the easiest way is to change all reward candidates
    """
    cons = [self.consStates[_] for _ in self.unknownCons]

    isAnUnsafeState = lambda s: any(s in conStates for conStates in cons)
    newRFuncs = map(lambda r: lambda s, a: r(s, a) - self.costOfQuery if isAnUnsafeState(s) else r(s, a), mdp.rFuncs)

    mdp.setReward(zip(newRFuncs, mdp.psi))


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
    self.rewardQueryAgent = GreedyConstructRewardAgent(mdp, 2, qi=True)
    self.featureQueryAgent = GreedyForSafetyAgent(mdp, consStates, goalStates, consProbs, improveSafePis=True)

  def updateFeats(self, newFreeCon=None, newLockedCon=None):
    JointUncertaintyQueryAgent.updateFeats(self, newFreeCon, newLockedCon)
    # need to update the feature partition of self.featureQueryAgent
    self.featureQueryAgent.updateFeats(newFreeCon, newLockedCon)

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

    # going to modify the transition function in place, so make a copy of mdp
    # for any unknown feature, with prob. pf, its transition goes through. Otherwise it's transited to a sink state
    # for any locked feature, it transits to a sink state with prob. 1
    # free features wouldn't pose any constraints
    mdp = copy.deepcopy(self.mdp)
    self.encodeConstraintIntoTransition(mdp)
    self.rewardQueryAgent.mdp = mdp

    # assume reward-set query has binary responses, so pose either one
    return self.rewardQueryAgent.findRewardSetQuery()[0]

  def findFeatureQuery(self, subsetCons=None):
    """
    use set-cover based algorithm and use the mean reward function (mdp.r does that)

    when safe policies exist, need to modify the original algorithm:
    computing the set structures by first removing safe dominating policies (set includeSafePolicies to True),
    that is, we want to minimize the number of queries to find *additional* dominating policies.
    """
    # recompute the set cover structure under the mean reward function (it will use self.mdp.r)
    self.featureQueryAgent.computePolicyRelFeats(recompute=True)
    self.featureQueryAgent.computeIISs(recompute=True)

    # after computing rel feats, check if it's empty. if so, nothing need to be queried.
    if len(self.featureQueryAgent.domPiFeats) == 0 or len(self.featureQueryAgent.iiss) == 0: return None

    return self.featureQueryAgent.findQuery(subsetCons=subsetCons)

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

      # we use the mdp with safety constraints encoded into the transition function
      mdpIfTrueReward = copy.deepcopy(self.rewardQueryAgent.mdp)
      mdpIfTrueReward.updatePsi(computePosteriorBelief(mdpIfTrueReward.psi, consistentRewards=rIndices))
      posteriorValueIfTrue = self.findConstrainedOptPi(activeCons=self.unknownCons, mdp=mdpIfTrueReward)['obj']

      mdpIfFalseReward = copy.deepcopy(self.rewardQueryAgent.mdp)
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
    queryAndEVOIs = []

    for query in queries:
      #fixme a bit awkward to change the representation of features
      queryAndEVOIs.append((query, self.computeEVOI(query)))

    if config.VERBOSE: print queryAndEVOIs

    optQueryAndEVOI = max(queryAndEVOIs, key=lambda _: _[1])

    if considerCost and optQueryAndEVOI[1] < self.costOfQuery:
      return None
    elif optQueryAndEVOI[0][1] is None:
      return None
    else:
      return optQueryAndEVOI[0]

  def findQuery(self):
    """
    compute the myopically optimal reward query vs feature query, pose the on that has larger EPU value
    """
    rewardQuery = ('R', self.findRewardQuery())
    featureQuery = ('F', self.findFeatureQuery())

    return self.selectQueryBasedOnEVOI([rewardQuery, featureQuery])


class JointUncertaintyBatchQueryAgent(JointUncertaintyQueryByMyopicSelectionAgent):
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None, costOfQuery=0):
    JointUncertaintyQueryByMyopicSelectionAgent.__init__(self, mdp, consStates, goalStates=goalStates,
                                                         consProbs=consProbs, costOfQuery=costOfQuery)

    # use a reward query agent that considers querying cost
    self.rewardQueryAgent = GreedyConstructRewardWithConsCostAgent(mdp, 2, consStates=consStates,
                                                                   costOfQuery=costOfQuery, qi=True)

  def findBatchQuery(self):
    """
    find one reward query and some feature queries to possibly query about
    safety constraints are encoded in the transition function
    """
    psiSupports = filter(lambda _: _ > 0, self.mdp.psi)

    # going to modify the transition function of the mdp used by rewardQueryAgent
    mdp = copy.deepcopy(self.mdp)

    # consider the possible responses to the queried features
    self.encodeConstraintIntoTransition(mdp)
    self.rewardQueryAgent.mdp = mdp
    # used for charging costOfQuery when visiting states with changed unknown features
    self.rewardQueryAgent.consStates = [self.consStates[_] for _ in self.unknownCons]

    qPi = self.rewardQueryAgent.findPolicyQuery()
    qR = self.rewardQueryAgent.findRewardSetQuery(qPi)

    qFeats = set()
    for pi in qPi:
      violatedCons = self.findViolatedConstraints(pi)
      qFeats = qFeats.union(violatedCons)

    # using set cover criterion to find the best feature query
    if len(qFeats) > 0: qFeat = self.findFeatureQuery(subsetCons=qFeats)
    else: qFeat = None

    queries = [('R', q) for q in qR if 0 < len(q) < len(psiSupports)] + [('F', qFeat)]

    # if the batch query has no positive evoi, stop querying
    eus = self.rewardQueryAgent.computeEUS(qPi, qR)
    priorValue = self.computeCurrentSafelyOptPiValue()
    batchQueryEVOI = eus - priorValue
    # count the cost of reward query
    if any(_[0] == 'R' for _ in queries): batchQueryEVOI -= self.costOfQuery
    # note: this is not exactly the definition of evoi since it could be negative. we counted the query cost!

    if config.VERBOSE: print 'evoi', eus, '-', priorValue, '=', batchQueryEVOI
    if batchQueryEVOI <= 1e-4: return None

    return queries

  def findQuery(self):
    # find an instance of good reward query + feature queries
    queries = self.findBatchQuery()

    if queries is None or len(queries) == 0: return None
    else: return self.selectQueryBasedOnEVOI(queries, considerCost=False)


class JointUncertaintyQueryBySamplingDomPisAgent(JointUncertaintyQueryAgent):
  """
  Sample a set of dominating policies according to their probabilities of being free and their values.
  Then query the features that would make them safely-optimal.
  """
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None, costOfQuery=0, heuristicID=0):
    JointUncertaintyQueryAgent.__init__(self, mdp, consStates, goalStates=goalStates, consProbs=consProbs,
                                        costOfQuery=costOfQuery)

    # initialize objectDomPiData to be None, will be computed in findQuery
    self.objectDomPiData = None

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

  def findDomPi(self):
    """
    (re)compute all dominating policies given reward and safety uncertainty
    and then sample one
    stored in self.dompis = [(dompi, weighted_prob)]
    """
    domPisData = []

    priorValue = self.computeCurrentSafelyOptPiValue()
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

        domPisDatum = self.DomPiData()
        domPisDatum.pi = domPi
        domPisDatum.optimizedRewards = rIndices
        domPisDatum.violatedCons = relFeats

        if self.heuristicID == 0:
          # we are going to query about rIndices and relFeatures
          # we regard them as batch queries and compute the possible responses
          safeProb = reduce(mul, [self.consProbs[feat] for feat in relFeats], 1)
          rPositiveValue = rewardPositiveConsAgent.computeValue(domPi)

          # at least (relFeats) feature queries and 1 reward-set query are needed
          weightedValue = safeProb * sumOfPsi * (rPositiveValue - priorValue)
          # punish it by the number of queries asked
          weightedValue -= self.costOfQuery * len(relFeats)
          # reward query cost
          if len(rIndices) < self.sizeOfRewards: weightedValue -= self.costOfQuery

          domPisDatum.weightedValue = weightedValue
        elif self.heuristicID == 1:
          domPisDatum.weightedValue = 1.0
        else:
          raise Exception('unknown heuristicID ' + str(self.heuristicID))

        domPisData.append(domPisDatum)

    if len(domPisData) > 0:
      self.objectDomPiData = max(domPisData, key=lambda datum: datum.weightedValue)

    if len(domPisData) == 0 or self.objectDomPiData.weightedValue <= 0:
      # no dompis to consider, or the value says nothing worth querying
      self.objectDomPiData = None
      return

    if config.VERBOSE: print 'chosen dom pi', self.objectDomPiData

  def attemptToFindQuery(self):
    """
    try to find a query without re-sampling domPi
    return None if can't do so
    """
    # if not assigned, compute it
    if self.objectDomPiData is None:
      if config.VERBOSE: print 'dompi not assigned'
      return None

    # if some relevant features are known-to-be-locked, we should find another dom pi
    if len(set(self.knownLockedCons).intersection(self.objectDomPiData.violatedCons)) > 0:
      if config.VERBOSE: print 'dompi not safe'
      return None

    # first consider reward queries
    consistentRewardIndices = self.computeConsistentRewardIndices(self.mdp.psi)
    # if any of the reward functions tha domPi optimizes is not possibly a true reward function
    if not set(self.objectDomPiData.optimizedRewards).issubset(consistentRewardIndices):
      if config.VERBOSE: print 'some optimized reward known to be false'
      return None
    # otherwise, try to find a reward query
    qReward = set(self.objectDomPiData.optimizedRewards).intersection(consistentRewardIndices)
    # make sure we have something to query about (not querying the whole conssitentRewardIndices set)
    if len(qReward) < len(consistentRewardIndices):
      return ('R', list(qReward))

    # then we consider feature queries
    unknownRelFeats = set(self.objectDomPiData.violatedCons).intersection(self.unknownCons)
    if len(unknownRelFeats) > 0:
      # pose a feature query with the largest p_f value
      qFeat = max(unknownRelFeats, key=lambda _: self.consProbs[_])
      return ('F', qFeat)

    # otherwise, we have nothing to query about
    if config.VERBOSE: print 'nothing to query'
    return None

  def findQuery(self):
    """
    sample some dominating policies, find the most useful query?
    :return: (query type, query)
    """
    query = self.attemptToFindQuery()
    if query is None:
      self.findDomPi()
      query = self.attemptToFindQuery()

    # possibly query is still None, in which case return None and stop querying
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

