import copy

import config
from algorithms.jointUncertaintyAgents import JointUncertaintyQueryByMyopicSelectionAgent
from algorithms.lp import computeValue, lpDualGurobi, jointUncertaintyMilp


class JointUncertaintyBatchQueryAgent(JointUncertaintyQueryByMyopicSelectionAgent):
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None, costOfQuery=0, qi=True):
    JointUncertaintyQueryByMyopicSelectionAgent.__init__(self, mdp, consStates, goalStates=goalStates,
                                                         consProbs=consProbs, costOfQuery=costOfQuery)
    # use query iteration
    self.qi = qi
    # a list of {'pi': ..., 'zC': ...} of each policy in the query
    self.querySolns = None

  def findPolicyQuery(self):
    """
    TODO only implemented for k = 2
    :return: a two policy query
    """
    # initial policy
    self.querySolns.append(firstPiSoln = lpDualGurobi(self.mdp, zeroConstraints=self.consStates, violationCost=self.costOfQuery))
    # the second policy
    self.findNextQuery()

  def findNextQuery(self):
    """
    in-place find the next policy
    :return:
    """
    self.querySolns.append(jointUncertaintyMilp(self.mdp, self.querySolns[0]['pi'], self.querySolns[0]['zC'],
                                                lockedFeatStates=[self.consStates[idx] for idx in self.knownLockedCons],
                                                unknownFeatStates=[self.consStates[idx] for idx in self.unknownCons],
                                                costOfQuery=self.costOfQuery))

  def queryIteration(self):
    pass

  def findRewardSetQuery(self, qPi=None):
    """
    :param qPi: provided if already computed
    :return: [indices of rewards being dominateded by policy, for policy in qPi]
    """
    if qPi is None: qPi = self.findPolicyQuery()

  def computeEUS(self):
    pass

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
    # don't consider immediate cost for query selection, so could select a query with EVOI < costOfQuery
    else: return self.selectQueryBasedOnEVOI(queries, considerCost=False)

