import copy

import config
from algorithms.jointUncertaintyAgents import JointUncertaintyQueryByMyopicSelectionAgent
from algorithms.lp import lpDualGurobi, jointUncertaintyMilp
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

  def computeZC(self, pi):
    """
    zC[i] = 1 if the i-th unknown feature is changed by pi, 0 otherwise.
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
    """
    Find the first policy in the policy query by optimizing the objective (V - cost of query)
    :param psi: if set, use reward belief psi. otherwise, under self.mdp.psi
    :return: the initial policy in the policy query
    """
    if psi is not None:
      mdp = copy.deepcopy(self.mdp)
      mdp.updatePsi(psi)
    else:
      mdp = self.mdp

    # this objective subtracts costs of violating unknown features
    return lpDualGurobi(mdp,
                        unknownStateCons=self.getUnknownFeatCons(),
                        violationCost=self.costOfQuery)['pi']

  def findNextPolicy(self, q):
    """
    Find the second policy given the previous one, using the modified MILP formulation.
    """
    assert len(q) == 1

    return jointUncertaintyMilp(self.mdp, q[0], self.computeZC(q[0]),
                                unknownFeatStates=self.getUnknownFeatCons(),
                                costOfQuery=self.costOfQuery)

  def computeEUS(self, qPi, qR=None):
    """
    note that this is not the exact objective which qPi tries to optimize, since we do not want to consider all possible
    realizations of changeability of unknown features.

    :return: E_{r, \Phi \in \Phi_\unknown} \max_{\pi \in (qPi \cap \Pi_\Phi) } V^\pi_r
    """
    piIndices = range(len(qPi))
    ret = 0
    for (r, rProb) in zip(self.mdp.rFuncs, self.mdp.psi):
      piValues = [self.computeValue(qPi[piIdx], r) for piIdx in piIndices]
      maxSafePiValue = max(piValues)
      ret += maxSafePiValue * rProb

    return ret

  def findBatchQuery(self):
    """
    find one reward query and some feature queries to possibly query about
    safety constraints are encoded in the transition function
    """
    priorValue = self.computeCurrentSafelyOptPiValue()

    # going to modify the transition function of the mdp
    self.encodeConstraintIntoTransition(self.mdp)

    # find the batch query
    qPi = self.findPolicyQuery()

    # find reward query
    # it's a 2-partition of reward functions, so pose either of them
    dominatedRewards = self.findRewardSetQuery(qPi)
    qR = dominatedRewards[0]
    # if it queries about all the reward functions or none of them, then no reward query is needed
    psiSupports = filter(lambda _: _ > 0, self.mdp.psi)
    if len(qR) == 0 or len(qR) >= len(psiSupports): qR = None

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
    else:
      qFeat = None

    # determine if continue querying is necessary
    eus = self.computeEUS(qPi)
    batchQueryEVOI = eus - priorValue
    # count the cost of reward query if posed
    if qR is not None: batchQueryEVOI -= self.costOfQuery
    if config.VERBOSE:
      print 'evoi', eus, '-', priorValue,
      if qR is not None: print '-', self.costOfQuery,
      print '=', batchQueryEVOI

    # note: this is not exactly the definition of evoi since it could be negative. we counted the query cost!
    if batchQueryEVOI <= 1e-4:
      return None
    else:
      # return selected reward query and feature query
      # if they ask about nothing (which is None), then don't include in the candidate queries
      queries = filter(lambda _: _[1] is not None, [('R', qR), ('F', qFeat)])
      return queries

  def findQuery(self):
    # find an instance of good reward query + feature queries
    # for now, recompute batch queries in each step
    currentMDP = copy.deepcopy(self.mdp)
    queries = self.findBatchQuery()
    # recover the current mdp FIXME better way to do this?
    self.mdp = currentMDP

    # in this case, not worth querying
    if queries is None or len(queries) == 0:
      return None
    else:
      # select one query from queries
      # don't consider immediate cost for query selection, so could select a query with EVOI < costOfQuery
      return self.selectQueryBasedOnEVOI(queries, considerCost=False)

      # or always first ask the reward query? empirically this is worse than selecting based on EVOI
      #return queries[0]

