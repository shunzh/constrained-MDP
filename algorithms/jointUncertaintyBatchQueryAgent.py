import copy

import config
from algorithms.jointUncertaintyAgents import JointUncertaintyQueryByMyopicSelectionAgent
from algorithms.lp import lpDualGurobi, jointUncertaintyMilp, computeValue
from algorithms.rewardQueryAgents import GreedyConstructRewardAgent
from util import powerset


class JointUncertaintyBatchQueryAgent(JointUncertaintyQueryByMyopicSelectionAgent,GreedyConstructRewardAgent):
  """
  We fix k to be 2 since it is sufficient to get enough information by considering what an approximately-optimal binary query asks about.
  It's a cons query agent and also a reward query agent, so deriving from two classes.
  """
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None, costOfQuery=0, qi=True):
    JointUncertaintyQueryByMyopicSelectionAgent.__init__(self, mdp, consStates, goalStates=goalStates,
                                                         consProbs=consProbs, costOfQuery=costOfQuery)
    GreedyConstructRewardAgent.__init__(self, mdp, 2, qi)

  def computeZC(self, pi):
    """
    convert relevant features to the format of zC
    """
    relFeats = self.findViolatedConstraints(pi)
    return [_ in relFeats for _ in range(len(self.unknownCons))]

  def computeValue(self, x, r=None):
    """
    compute value of policy - cost of querying
    """
    if r is None: r = self.mdp.r
    return GreedyConstructRewardAgent.computeValue(self, x, r) - self.costOfQuery * sum(self.computeZC(x))

  def findOptPolicyUnderMeanRewards(self, psi=None):
    if psi is not None:
      mdp = copy.deepcopy(self.mdp)
      mdp.updatePsi(psi)
    else:
      mdp = self.mdp

    return lpDualGurobi(mdp, zeroConstraints=self.getLockedFeatCons(), unknownStateCons=self.getUnknownFeatCons(),
                        violationCost=self.costOfQuery)['pi']

  def findNextPolicy(self, q):
    """
    find the second policy given the previous one
    """
    assert len(q) == 1
    return jointUncertaintyMilp(self.mdp, q[0], self.computeZC(q[0]),
                                zeroConstraints=self.getLockedFeatCons(), unknownFeatStates=self.getUnknownFeatCons(),
                                costOfQuery=self.costOfQuery)

  def computeEUS(self, qPi, qR=None):
    """
    FIXME qR is dummy, just to be consistent with the func signature
    note that this is not the exact objective which qPi tries to optimize, since we do not want to consider all possible
    realizations of changeability of unknown features.

    :return: E_{r, \Phi \in \Phi_\unknown} \max_{\pi \in (qPi \cap \Pi_\Phi) } V^\pi_r
    """
    piIndices = range(len(qPi))

    qPiRelFeats = [self.findViolatedConstraints(qPi[piIdx]) for piIdx in piIndices]

    ret = 0
    for (r, rProb) in zip(self.mdp.rFuncs, self.mdp.psi):
      piValues = [self.computeValue(qPi[piIdx], r) for piIdx in piIndices]
      for assumedFreeFeats in powerset(self.unknownCons):
        safePiIndices = filter(lambda piIdx: set(qPiRelFeats[piIdx]).issubset(assumedFreeFeats), piIndices)

        if len(safePiIndices) > 0:
          assumedLockedFeats = set(self.unknownCons) - set(assumedFreeFeats)
          featProb = self.probFeatsBeingFree(assumedFreeFeats) * self.probFeatsBeingLocked(assumedLockedFeats)

          maxSafePiValue = max(piValues[idx] for idx in safePiIndices)
          ret += maxSafePiValue * featProb * rProb

    return ret

  def findBatchQuery(self):
    """
    find one reward query and some feature queries to possibly query about
    safety constraints are encoded in the transition function
    """
    queries = []

    # going to modify the transition function of the mdp used by rewardQueryAgent
    mdp = copy.deepcopy(self.mdp)

    # consider the possible responses to the queried features
    self.encodeConstraintIntoTransition(mdp)
    qPi = self.findPolicyQuery()

    # find reward query
    # it's a 2-partition of reward functions, so pose either of them
    qR = self.findRewardSetQuery(qPi)[0]
    # if it queries about all the reward functions or none of them, then no reward query is needed
    psiSupports = filter(lambda _: _ > 0, self.mdp.psi)
    if 0 < len(qR) < len(psiSupports):
      queries.append(('R', qR))

    # find feat query
    qFeats = set()
    for pi in qPi:
      violatedCons = self.findViolatedConstraints(pi)
      qFeats = qFeats.union(violatedCons)
    # using set cover criterion to find the best feature query
    # we only consider querying about one of the features tha are relevant to policies in qPi
    # because others are not worth querying because of their costs? (although they may contribute to finding new safe policies)
    if len(qFeats) > 0:
      qFeat = self.findFeatureQuery(subsetCons=qFeats)
      queries.append(('F', qFeat))

    # determine if continue querying is necessary
    eus = self.computeEUS(qPi)
    priorValue = self.computeCurrentSafelyOptPiValue()
    batchQueryEVOI = eus - priorValue
    # count the cost of reward query
    if qR is None: batchQueryEVOI -= self.costOfQuery
    if config.VERBOSE: print 'evoi', eus, '-', priorValue, '=', batchQueryEVOI

    # note: this is not exactly the definition of evoi since it could be negative. we counted the query cost!
    if batchQueryEVOI <= 1e-4: return None
    else: return queries

  def findQuery(self):
    # find an instance of good reward query + feature queries
    # for now, recompute batch queries in each step
    queries = self.findBatchQuery()

    # in this case, not worth querying
    if queries is None: return None

    # select one query from queries
    # don't consider immediate cost for query selection, so could select a query with EVOI < costOfQuery
    else: return self.selectQueryBasedOnEVOI(queries, considerCost=False)

