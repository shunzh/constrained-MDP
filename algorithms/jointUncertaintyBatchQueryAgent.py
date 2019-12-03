import copy

import config
from algorithms.jointUncertaintyAgents import JointUncertaintyQueryByMyopicSelectionAgent
from algorithms.lp import lpDualGurobi, jointUncertaintyMilp, computeValue
from algorithms.rewardQueryAgents import GreedyConstructRewardAgent


class JointUncertaintyBatchQueryAgent(JointUncertaintyQueryByMyopicSelectionAgent,GreedyConstructRewardAgent):
  """
  We fix k to be 2 since it is sufficient to get enough information by considering what an approximately-optimal binary query asks about.
  It's a cons query agent and also a reward query agent, so deriving from two classes.
  """
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None, costOfQuery=0, qi=True):
    JointUncertaintyQueryByMyopicSelectionAgent.__init__(self, mdp, consStates, goalStates=goalStates,
                                                         consProbs=consProbs, costOfQuery=costOfQuery)
    GreedyConstructRewardAgent.__init__(self, mdp, 2, qi)

  def getLockedFeatCons(self):
    return [self.consStates[idx] for idx in self.knownLockedCons]

  def getUnknownFeatCons(self):
    return [self.consStates[idx] for idx in self.unknownCons]

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
    return GreedyConstructRewardAgent.computeValue(x, r) - self.costOfQuery * sum(self.computeZC(x))

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
    consStates = [self.consStates[_] for _ in self.unknownCons]

    qPi = self.findPolicyQuery()
    qR = self.findRewardSetQuery(qPi)

    qFeats = set()
    for pi in qPi:
      violatedCons = self.findViolatedConstraints(pi)
      qFeats = qFeats.union(violatedCons)

    # using set cover criterion to find the best feature query
    # we only consider querying about one of the features tha are relevant to policies in qPi
    # because others are not worth querying because of their costs? (although they may contribute to finding new safe policies)
    if len(qFeats) > 0: qFeat = self.findFeatureQuery(subsetCons=qFeats)
    else: qFeat = None

    # don't query about all or none of the reward candidates
    queries = [('R', q) for q in qR if 0 < len(q) < len(psiSupports)]\
              + [('F', qFeat)]

    # if the batch query has no positive evoi, stop querying
    eus = self.computeEUS(qPi, qR)
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
    # for now, recompute batch queries in each step
    queries = self.findBatchQuery()

    if queries is None or len(queries) == 0: return None
    # don't consider immediate cost for query selection, so could select a query with EVOI < costOfQuery
    else: return self.selectQueryBasedOnEVOI(queries, considerCost=False)

