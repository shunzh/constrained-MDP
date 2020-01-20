import copy
import math
import random
import time
from itertools import combinations

import config
from algorithms import lp
from algorithms.consQueryAgents import ConsQueryAgent, NOTEXIST, EXIST
from algorithms.setcover import coverFeat, removeFeat, killSupersets, numOfSetsContainFeat
from operator import mul

from util import powerset

class InitialSafePolicyAgent(ConsQueryAgent):
  def __init__(self, mdp, consStates, goalStates, consProbs=None, costOfQuery=0, knownFreeCons=(), knownLockedCons=(), improveSafePis=False):
    """
    :param costOfQuery: default cost of query is 1 unit
    """
    ConsQueryAgent.__init__(self, mdp, consStates, goalStates, consProbs=consProbs,
                            knownFreeCons=knownFreeCons, knownLockedCons=knownLockedCons)

    self.costOfQuery = costOfQuery
    self.improveSafePis = improveSafePis
    # counter for the number of queries asked
    self.numOfAskedQueries = 0

  def updateFeats(self, newFreeCon=None, newLockedCon=None):
    # change partition of features
    ConsQueryAgent.updateFeats(self, newFreeCon, newLockedCon)

    # udpate the set cover structure
    # recompute dom pi and iiss every time if we're doing early stopping
    if config.earlyStop is not None:
      # delete corresponding attributes and re-compute
      if hasattr(self, 'domPiFeats'):
        delattr(self, 'domPiFeats')
        self.computePolicyRelFeats()

      if hasattr(self, 'iiss'):
        delattr(self, 'iiss')
        self.computeIISs()
    else:
      if hasattr(self, 'relFeats'):
        # update relFeats to only contain unknown features
        self.relFeats = set(self.relFeats).intersection(self.unknownCons)

    self.numOfAskedQueries += 1

  def safePolicyExist(self, freeCons=None):
    """
    Safe policy known to exist
    at least one dom pi's relevant features are all free
    """
    # some dom pi's relevant features are all free
    if freeCons is None:
      freeCons = self.knownFreeCons

    if config.earlyStop is not None and hasattr(self, 'domPiFeats'):
      # if we have rel feat, simply check whether we covered all rel feats of any dom pi
      # this can only be used when earlyStop is not used, so we find exact domPiFeats
      return any(len(set(relFeats) - set(freeCons)) == 0 for relFeats in self.domPiFeats)
    else:
      # for some simple heuristics, it's not fair to ask them to precompute dompis (need to run a lot of LP)
      # so we try to solve the lp problem once here
      # see whether the lp is feasible if we assume all other features are locked
      return self.findConstrainedOptPi(set(self.unknownCons) - set(freeCons))['feasible']

  def safePolicyNotExist(self, lockedCons=None):
    """
    True is no safe policy known to exist
    there are some locked features in all dom pis
    """
    if lockedCons is None:
      lockedCons = self.knownLockedCons

    if config.earlyStop is not None and hasattr(self, 'piRelFeats'):
      return all(len(set(relFeats).intersection(lockedCons)) > 0 for relFeats in self.domPiFeats)
    else:
      # by only imposing these constraints, see whether the lp problem is infeasible
      return not self.findConstrainedOptPi(lockedCons)['feasible']

  def checkSafePolicyExists(self):
    """
    None if don't know, otherwise return exists or notExist
    """
    if self.improveSafePis: return None
    elif self.safePolicyExist(): return EXIST
    elif self.safePolicyNotExist(): return NOTEXIST
    else: return None

  def computePolicyRelFeats(self, recompute=False):
    """
    Compute relevant features of dominating policies.
    If the relevant features of any dominating policy are all free, then safe policies exist.
    Put in another way, if all dom pis has at least one locked relevant feature, then safe policies do not exist.

    This can be O(2^|relevant features|), depending on the implementation of findDomPis
    """
    # check whether this is already computed
    if hasattr(self, 'domPiFeats') and not recompute: return

    relFeats, domPis = self.findRelevantFeaturesAndDomPis()
    domPiFeats = []
    domPiFeatsAndValues = {}

    for domPi in domPis:
      feats = self.findViolatedConstraints(domPi)
      # if this is a known-to-be-safe dom pi and we aim to improve safe policies,
      # don't add this to the set cover structure
      if len(feats) == 0 and self.improveSafePis: continue

      domPiFeats.append(feats)
      # FIXME it may be easier to store the values when the dom pis are computed. recomputing here.
      domPiFeatsAndValues[tuple(feats)] = self.computeValue(domPi)

    self.domPiFeats = killSupersets(domPiFeats)
    self.domPiFeatsAndValues = domPiFeatsAndValues
    self.relFeats = relFeats # all relevant features

  def computeIISs(self, recompute=False):
    """
    Compute IISs by looking at relevant features of dominating policies.

    eg. (1 and 2) or (3 and 4) --> (1 or 3) and (1 or 4) and (2 or 3) and (2 or 4)
    """
    # we first need relevant features
    if hasattr(self, 'iiss') and not recompute: return

    if not hasattr(self, 'piRelFeats'):
      self.computePolicyRelFeats()

    iiss = [set()]
    # essentially convert DNF to CNF
    # incrementally consider more relFeats
    for relFeats in self.domPiFeats:
      iiss = [set(iis).union({relFeat}) for iis in iiss for relFeat in relFeats]
      # kill duplicates in each set
      iiss = killSupersets(iiss)

    self.iiss = iiss

  def computeIISsBruteForce(self):
    """
    DEPRECATED not an efficient way to compute IISs.

    If all IIS contain at least one free feature, then safe policies exist.

    a brute force way to find all iiss and return their indicies
    can be O(2^|unknown features|)
    """
    unknownConsPowerset = powerset(self.unknownCons)
    feasible = {}
    iiss = []

    for cons in unknownConsPowerset:
      # if any subset is already infeasible, no need to check this set. it's definitely infeasible and not an iis
      if len(cons) > 0 and any(not feasible[subset] for subset in combinations(cons, len(cons) - 1)):
        feasible[cons] = False
        continue

      # find if the lp is feasible by posing cons
      sol = self.findConstrainedOptPi(cons)
      feasible[cons] = sol['feasible']

      if len(cons) == 0 and not feasible[cons]:
        # no iiss in this case. problem infeasible
        return []

      # if it is infeasible and all its subsets are feasible (only need to check the subsets with one less element)
      # then it's an iis
      if not feasible[cons] and all(feasible[subset] for subset in combinations(cons, len(cons) - 1)):
        iiss.append(cons)

    self.iiss = iiss

  def safePolicyValue(self):
    sol = self.findConstrainedOptPi(set(self.consIndices) - set(self.knownFreeCons))
    if sol['feasible']:
      return sol['obj']
    else:
      # for comparison, the return is 0 when no safe policies exist
      return 0

  def getProbOfExistenceOfSafePolicies(self, lockedCons, freeCons):
    """
    Compute the probability that safe policies exist using dominating policies (the best way?)

    lockedCons, freeCons: The set of locked and free features.
      They might be different from the ones confirmed by querying.
      These are hypothetical ones just to compute the corresponding prob.
    """
    result = 0

    def pf(con):
      if con in lockedCons: return 0
      elif con in freeCons: return 1
      else: return self.consProbs[con]

    assert hasattr(self, 'domPiFeats')

    # two ways to compute the probs. either 2^|relFeats| or 2^|domPis|
    # so see which one is smaller
    if len(self.relFeats) < len(self.domPiFeats):
      allSubsetsOfRelFeats = powerset(self.relFeats)
      for freeSubset in allSubsetsOfRelFeats:
        if self.safePolicyExist(freeCons = list(freeSubset) + list(self.knownFreeCons)):
          prob = self.probFeatsBeingFree(freeSubset) * self.probFeatsBeingLocked(set(self.relFeats) - set(freeSubset))
          result += prob
    else:
      for k in range(1, len(self.domPiFeats) + 1):
        sign = 1 if k % 2 == 1 else -1
        for domPiFeatsSubset in combinations(self.domPiFeats, k):
          domPiFeatsSubsetLists = map(lambda _: list(_), domPiFeatsSubset)
          unionOfFeats = set(sum(domPiFeatsSubsetLists, []))
          result += sign * reduce(mul, map(pf, unionOfFeats), 1)

    return result


class GreedyForSafetyAgent(InitialSafePolicyAgent):
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None, useIIS=True, useRelPi=True,
               optimizeValue=False, heuristicID=0, improveSafePis=False, k=1):
    """
    :param consStates: the set of states that should not be visited
    :param consProbs: the probability that the corresponding constraint is free, None if adversarial setting
    :param useIIS: set cover on IIS. False by default and only enable it by explicitly setting true
    :param useRelPi: set cover on relevant features
    :param heuristicID: an hack for trying different heuristics
    :param optimizeValue: True if hoping to find a safe policy with higher values
    :param improveSafePis: True if we want to include safe dominating policies if they exist
      they should not exist if no safe policies exist. Otherwise will remove them if False.
    """
    InitialSafePolicyAgent.__init__(self, mdp, consStates, goalStates, consProbs=consProbs, improveSafePis=improveSafePis)

    self.useIIS = useIIS
    self.useRelPi = useRelPi

    self.optimizeValue = optimizeValue
    self.heuristicID = heuristicID
    self.k = k

    # find all IISs without knowing any locked or free cons
    if self.useRelPi:
      self.computePolicyRelFeats()

    if self.useIIS:
      self.computeIISs()

    if self.optimizeValue:
      self.computeFeatureValue()

  def computeFeatureValue(self):
    """
    Compute how much each feature worth for finding a high-valued safe policy. (w in the report)
    saves to self.featureVals
    """
    # otherwise cannot compute rel feat values
    assert self.useRelPi

    n = len(self.domPiFeats) # number of dominating policies
    d = len(self.relFeats) # number of relevant features

    A = [[1 if self.relFeats[j] in self.domPiFeats[i] else 0 for j in range(d)] for i in range(n)]
    assert all(tuple(self.domPiFeats[i]) in self.domPiFeatsAndValues.keys() for i in range(n))
    b = [self.domPiFeatsAndValues[tuple(self.domPiFeats[i])] for i in range(n)]
    weights = lp.linearRegression(A, b)

    self.featureVals = {}
    for i in range(d): self.featureVals[self.relFeats[i]] = weights[i]
    print 'feat vals', filter(lambda _: _[1] > 1e-4, self.featureVals.items())

  def updateFeats(self, newFreeCon=None, newLockedCon=None):
    # this just add to the list of known free and locked features
    InitialSafePolicyAgent.updateFeats(self, newFreeCon, newLockedCon)

    if newFreeCon is not None:
      if self.useIIS:
        self.iiss = coverFeat(newFreeCon, self.iiss)
      if self.useRelPi:
        self.domPiFeats = removeFeat(newFreeCon, self.domPiFeats)

    if newLockedCon is not None:
      if self.useIIS:
        self.iiss = removeFeat(newLockedCon, self.iiss)
      if self.useRelPi:
        self.domPiFeats = coverFeat(newLockedCon, self.domPiFeats)

  def findQuery(self, subsetCons=None):
    """
    :param subsetCons: only consider intersection of this set if not none
    :return: the next feature to query by greedily cover the most number of sets
    return None if no more features are needed or nothing left to query about

    heuristics are inspired by
    Golovin, Daniel, and Andreas Krause. "Adaptive submodularity: Theory and applications in active learning and stochastic optimization."
    Journal of Artificial Intelligence Research 42 (2011): 427-486.
    """
    answerFound = self.checkSafePolicyExists()
    if answerFound != None: return answerFound

    # make sure the constraints that are already queried are not going to be queried again
    relFeats = self.relFeats
    if subsetCons is not None: relFeats = set(relFeats).intersection(subsetCons)
    if len(relFeats) == 0: return None

    # find the maximum frequency constraint weighted by the probability
    score = {}

    # compute quantities used by some heuristics
    if self.heuristicID == 3:
      probSafePiExistWhenFree = {con: self.getProbOfExistenceOfSafePolicies(self.knownLockedCons, self.knownFreeCons + [con]) for con in relFeats}
      probSafePiExistWhenLocked = {con: self.getProbOfExistenceOfSafePolicies(self.knownLockedCons + [con], self.knownFreeCons) for con in relFeats}

    if self.heuristicID == 1:
      probSafePiExist = self.getProbOfExistenceOfSafePolicies(self.knownLockedCons, self.knownFreeCons)

    for con in relFeats:
      if self.optimizeValue:
        # try to optimize values of the safe policies
        # we need IIS when trying to optimize the values of policies
        score[con] = self.consProbs[con] * (self.costOfQuery - self.featureVals[con]) / len(filter(lambda _: con in _, self.iiss)) \
                   + (1 - self.consProbs[con]) * self.costOfQuery / len(filter(lambda _: con in _, self.domPiFeats))
      else:
        # only aim to find a safe policy (regardless of its value)
        if self.heuristicID == 0:
          # original heuristic, h_{SC}
          #score[con] = self.consProbs[con] * iisNumWhenFree + (1 - self.consProbs[con]) * relNumWhenLocked
          score[con] = 0
          if self.useIIS:
            score[con] += self.useIIS * self.consProbs[con] * numOfSetsContainFeat(con, self.iiss) / len(self.iiss)
          if self.useRelPi:
            score[con] += self.useRelPi * (1 - self.consProbs[con]) * numOfSetsContainFeat(con, self.domPiFeats) / len(self.domPiFeats)
        elif self.heuristicID == 1:
          score[con] = self.consProbs[con] * probSafePiExist * numOfSetsContainFeat(con, self.iiss)\
                     + (1 - self.consProbs[con]) * (1 - probSafePiExist) * numOfSetsContainFeat(con, self.domPiFeats)
        elif self.heuristicID == 2:
          estimateCoverElems = lambda s, prob: min(1.0 * len(s) / (prob(nextCon) * numOfSetsContainFeat(nextCon, s) + 1e-4) for nextCon in relFeats)
          freeProb = lambda _: self.consProbs[_]
          lockedProb = lambda _: 1 - self.consProbs[_]

          score[con] = (self.consProbs[con] * numOfSetsContainFeat(con, self.iiss) / len(self.iiss)
                       * (probSafePiExist * estimateCoverElems(self.iiss, freeProb))
                     + (1 - self.consProbs[con]) * numOfSetsContainFeat(con, self.domPiFeats) / len(self.domPiFeats)
                       * (1 - probSafePiExist) * estimateCoverElems(self.domPiFeats, lockedProb))
        elif self.heuristicID == 3:
          # this heuristic uses coverage ratio estimate
          estimateCoverElems = lambda s, prob: min(1.0 * len(s) / (prob(nextCon) * numOfSetsContainFeat(nextCon, s) + 1e-4) for nextCon in relFeats)
          # useful locally
          freeProb = lambda _: self.consProbs[_]
          lockedProb = lambda _: 1 - self.consProbs[_]

          score[con] = self.consProbs[con] * (probSafePiExistWhenFree[con] * estimateCoverElems(coverFeat(con, self.iiss), freeProb) +
                                              (1 - probSafePiExistWhenFree[con]) * estimateCoverElems(removeFeat(con, self.domPiFeats), lockedProb))\
                     + (1 - self.consProbs[con]) * (probSafePiExistWhenLocked[con] * estimateCoverElems(removeFeat(con, self.iiss), freeProb) +
                                                    (1 - probSafePiExistWhenLocked[con]) * estimateCoverElems(coverFeat(con, self.domPiFeats), lockedProb))
          # minimize this objective
          score[con] = -score[con]
        else:
          raise Exception('unknown heuristicID')

      # semantically, score estimates the number of queries, so count the current feature
      #score[con] += 1

    # to understand the behavior
    return max(score.iteritems(), key=lambda _: _[1])[0]

  def findKFeatureQuery(self):
    """
    Construct a k-feature batch query using greedy construction.

    Chen, Yuxin, and Andreas Krause. "Near-optimal Batch Mode Active Learning and Adaptive Submodular Optimization."
    ICML, 2013.
    """
    #FIXME not reusing code in findQuery, only implementing heuristic 0
    assert self.useIIS and self.useRelPi

    query = []

    for queryIdx in range(min(self.k, len(self.relFeats))):
      # create set structures based on all possible responses to previously selected features in the same batch
      iisUncoveredProbs = {}
      relFeatUncoveredProbs = {}

      for iis in self.iiss:
        iisUncoveredProbs[iis] = self.probFeatsBeingLocked(set(iis).intersection(query))
      for feats in self.domPiFeats:
        relFeatUncoveredProbs[feats] = self.probFeatsBeingFree(set(feats).intersection(query))

      score = {}
      for con in self.relFeats:
        score[con] = self.consProbs[con] * sum(iisUncoveredProbs[iis] for iis in self.iiss if con in iis) / len(self.iiss)\
                   + (1 - self.consProbs[con]) * sum(relFeatUncoveredProbs[feats] for feats in self.domPiFeats if con in feats) / len(self.domPiFeats)
      query.append(max(self.relFeats, key=lambda _: score[_]))

    return query


class DomPiHeuForSafetyAgent(InitialSafePolicyAgent):
  """
  This uses dominating policies. It first finds the dominating policy that has the largest probability being free.
  Then query about the most probable unknown feature in the relevant features of the policy.
  """
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None, optimizeValue=False):
    InitialSafePolicyAgent.__init__(self, mdp, consStates, goalStates, consProbs)

    self.computePolicyRelFeats()
    self.optimizeValue = optimizeValue

  def findQuery(self):
    answerFound = self.checkSafePolicyExists()
    if answerFound != None: return answerFound

    relFeats = self.relFeats

    updatedConsProbs = {}
    # find the most probable policy
    # update the cons prob to make it easier
    for i in self.consIndices:
      if i in self.knownLockedCons: updatedConsProbs[i] = 0
      elif i in self.knownFreeCons: updatedConsProbs[i] = 1
      elif i in self.unknownCons: updatedConsProbs[i] = self.consProbs[i]
      else: raise Exception('uncategorized feature, should not happen')

    # find the policy that has the largest probability to be feasible
    feasibleProb = lambda relFeats: reduce(mul,
                                           map(lambda _: updatedConsProbs[_], relFeats),
                                           1)

    maxProbPiRelFeats = max(self.domPiFeats, key=feasibleProb)

    # now query about unknown features in the most probable policy's relevant features
    featsToConsider = set(relFeats).intersection(maxProbPiRelFeats)
    # the probability is less than 1. so there must be unknown features to consider
    assert len(featsToConsider) > 0
    return max(featsToConsider, key=lambda _: self.consProbs[_])


class MaxProbSafePolicyExistAgent(InitialSafePolicyAgent):
  """
  Find the feature that, after querying, the expected probability of finding a safe poicy / no safe policies exist is maximized.
  """
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None, tryFeasible=True, tryInfeasible=True):
    InitialSafePolicyAgent.__init__(self, mdp, consStates, goalStates, consProbs)

    self.tryFeasible = tryFeasible
    self.tryInfeasible = tryInfeasible

    # need domPis for query
    self.computePolicyRelFeats()

  def findQuery(self):
    answerFound = self.checkSafePolicyExists()
    if answerFound != None: return answerFound

    relFeats = self.relFeats

    # the probability that either
    termProbs = {}
    for con in relFeats:
      termProbs[con] = 0
      # the prob that safe policies exist when con is free
      if self.tryFeasible:
        probExistWhenFree = self.getProbOfExistenceOfSafePolicies(self.knownLockedCons, self.knownFreeCons + [con])
        termProbs[con] += self.tryFeasible * self.consProbs[con] * probExistWhenFree

      if self.tryInfeasible:
        probNotExistWhenLocked = 1 - self.getProbOfExistenceOfSafePolicies(self.knownLockedCons + [con], self.knownFreeCons)
        termProbs[con] += self.tryInfeasible * (1 - self.consProbs[con]) * probNotExistWhenLocked

    # there should be unqueried features
    assert len(termProbs) > 0

    return max(termProbs.iteritems(), key=lambda _: _[1])[0]


class DescendProbQueryForSafetyAgent(InitialSafePolicyAgent):
  """
  Return the unknown feature that has the largest (or smallest) probability of being changeable.
  """
  def findQuery(self):
    answerFound = self.checkSafePolicyExists()
    if answerFound != None: return answerFound

    unknownCons = self.unknownCons
    assert len(unknownCons) > 0
    return max(unknownCons, key=lambda con: self.consProbs[con])


class RandomQueryAgent(InitialSafePolicyAgent):
  def findQuery(self):
    answerFound = self.checkSafePolicyExists()
    if answerFound != None: return answerFound

    unknownCons = self.relFeats
    assert len(unknownCons) > 0
    return random.choice(unknownCons)


class OptQueryForSafetyAgent(InitialSafePolicyAgent):
  """
  Find the opt query by dynamic programming. Its O(2^|\Phi|).
  """
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None, optimizeLocked=True, optimizeFree=True):
    InitialSafePolicyAgent.__init__(self, mdp, consStates, goalStates, consProbs)

    # need domPis for query
    self.computePolicyRelFeats()

    # (locked, free) -> [(query, expected value)]
    self.optQs = {}
    # the set of cons imposing which we have a safe policy
    self.freeBoundary = []
    # the set of cons imposing which we don't have a safe policy for sure
    self.lockedBoundary = []

    self.initRelFeats = copy.copy(self.relFeats)

    # fixme do not need terminal cost for now
    self.lockedTerminalCost = self.freeTerminalCost = 0
    #self.lockedTerminalCost = 1 if optimizeLocked else 0
    #self.freeTerminalCost = 1 if optimizeFree else 0

    # this is computed in advance
    self.computeOptQueries()

  def getQueryAndValue(self, locked, free, allValues=False):
    if any(set(locked).issuperset(lockedB) for lockedB in self.lockedBoundary):
      return (NOTEXIST, self.lockedTerminalCost)
    elif any(set(free).issuperset(freeB) for freeB in self.freeBoundary):
      return (EXIST, self.freeTerminalCost)
    elif not (frozenset(locked), frozenset(free)) in self.optQs:
      return None
    else:
      if not allValues:
        return min(self.optQs[frozenset(locked), frozenset(free)], key=lambda _: _[1])
      else:
        # if allValues, return the whole list of q function
        return self.optQs[frozenset(locked), frozenset(free)]

  def setQueryAndValue(self, locked, free, minNums):
    # keep the whole `q function' for debugging
    self.optQs[frozenset(locked), frozenset(free)] = minNums

  def computeOptQueries(self):
    """
    f(\phi_l, \phi_f) =
      0, if safePolicyExist(\phi_f) or self.safePolicyNotExist(\phi_l)
      min_\phi p_f(\phi) f(\phi_l, \phi_f + {\phi}) + (1 - p_f(\phi)) f(\phi_l + {\phi}, \phi_f), o.w.

    Boundaries condition:
      \phi_l is not a superset of any iis, \phi_f is not a superset of rel feats of any dom pi, otherwise 0 for sure
    """
    consPowerset = list(powerset(self.relFeats))

    # free/locked cons that are not supersets of elements on their boundaries
    # that is, we can't determine if safe policy exsits if these elements are known to be locked/free
    admissibleFreeCons = []
    admissibleLockedCons = []
    # the set of (lockedCons, freeCons) to evaluate the optimal queries
    # it's the cross product of the two sets above, excluding free and locked cons that share elements
    admissibleCons = []

    for lockedCons in consPowerset:
      if self.safePolicyNotExist(lockedCons=lockedCons):
        # make sure not elements in boundary is a superset of another element
        if not any(set(lockedCons).issuperset(lockedB) for lockedB in self.lockedBoundary):
          self.lockedBoundary.append(lockedCons)
      else: admissibleLockedCons.append(lockedCons)

    for freeCons in consPowerset:
      if self.safePolicyExist(freeCons=freeCons):
        # similarly
        if not any(set(freeCons).issuperset(freeB) for freeB in self.freeBoundary):
          self.freeBoundary.append(freeCons)
      else: admissibleFreeCons.append(freeCons)

    if config.DEBUG:
      print 'locked', self.lockedBoundary
      print 'free', self.freeBoundary

    for lockedCons in admissibleLockedCons:
      for freeCons in admissibleFreeCons:
        # any cons should not be known to be both free and locked
        if set(lockedCons).isdisjoint(set(freeCons)):
          admissibleCons.append((lockedCons, freeCons))

    # make sure all terms on the RHS (of def of f above) are evaluated
    readyToEvaluate = lambda l, f: all(self.getQueryAndValue(l, set(f).union({con})) != None \
                                       and self.getQueryAndValue(set(l).union({con}), f) != None \
                                       for con in set(self.relFeats) - set(l) - set(f))

    # keep the sets of cons that are ready to evaluate in the next iteration
    readyToEvalSet = []
    for (lockedCons, freeCons) in admissibleCons:
      if readyToEvaluate(lockedCons, freeCons): readyToEvalSet.append((lockedCons, freeCons))

    # keep fill out the values of optQs within boundary
    # whenever filled out
    while len(readyToEvalSet) > 0:
      if config.DEBUG: print len(readyToEvalSet), 'need to be evaluated'

      (lockedCons, freeCons) = readyToEvalSet.pop()

      unknownCons = set(self.relFeats) - set(lockedCons) - set(freeCons)

      # evaluate all candidate con and compute their minimum number of queries
      minNums = [(con,
                  self.consProbs[con] * self.getQueryAndValue(lockedCons, set(freeCons).union({con}))[1]\
                  + (1 - self.consProbs[con]) * self.getQueryAndValue(set(lockedCons).union({con}), freeCons)[1]\
                  + 1)
                 for con in unknownCons]

      # pick the tuple that has the minimum obj value after querying
      self.setQueryAndValue(lockedCons, freeCons, minNums)

      # add neighbors that ready to evaluate to readToEvalSet
      readyToEvalSet += filter(lambda (l, f): self.getQueryAndValue(l, f) == None and readyToEvaluate(l, f),
                               [(set(lockedCons) - {cons}, freeCons) for cons in lockedCons] +\
                               [(lockedCons, set(freeCons) - {cons}) for cons in freeCons])

  def findQuery(self):
    # we only care about the categories of rel feats
    relLockedCons = set(self.knownLockedCons).intersection(self.initRelFeats)
    relFreeCons = set(self.knownFreeCons).intersection(self.initRelFeats)

    qAndV = self.getQueryAndValue(relLockedCons, relFreeCons)
    assert qAndV != None

    if config.VERBOSE:
      print 'query and value', self.getQueryAndValue(relLockedCons, relFreeCons, allValues=True)

    return qAndV[0]


class OracleSafetyAgent(InitialSafePolicyAgent):
  """
  It knows the true partitions of features.
  So it can ask about the minimum number of features.
  """
  def __init__(self, mdp, consStates, trueFreeFeatures, goalStates=(), consProbs=None):
    InitialSafePolicyAgent.__init__(self, mdp, consStates, goalStates, consProbs)

    self.queries = []
    self.answer = None

    self.trueFreeFeatures = trueFreeFeatures
    self.trueLockedFeatures = set(self.consIndices) - set(trueFreeFeatures)

    self.safePolicyIndeedExist = self.safePolicyExist(trueFreeFeatures)

    # we know the answer before querying, set it here
    if self.safePolicyIndeedExist:
      self.answer = EXIST
    else:
      self.answer = NOTEXIST

    self.computePolicyRelFeats()
    self.computeIISs()

    # depend on whether we find the true dom pis
    if config.earlyStop is None:
      self.computeExactQueries()
    else:
      self.computeApproximateQueries()

  def computeExactQueries(self):
    if self.safePolicyIndeedExist:
      freePiFeats = filter(lambda _: set(self.trueFreeFeatures).issuperset(_), self.domPiFeats)
      assert len(freePiFeats) > 0
      self.queries = min(freePiFeats, key=lambda _: len(_))
    else:
      lockedIISs = filter(lambda _: set(self.trueLockedFeatures).issuperset(_), self.iiss)
      assert len(lockedIISs) > 0
      self.queries = min(lockedIISs, key=lambda _: len(_))

    self.queries = list(self.queries)

  def computeApproximateQueries(self):
    """
    We do not have the computation power to find all domPis and IISs.
    So we greedily cover tree free/locked features
    :return:
    """
    if self.safePolicyIndeedExist:
      if len(self.iiss) == 0:
        self.queries = []
      else:
        query = max(set(self.trueFreeFeatures) - set(self.knownFreeCons), key=lambda con: numOfSetsContainFeat(con, self.iiss))
        self.queries = [query]
    else:
      if len(self.domPiFeats) == 0:
        self.queries = []
      else:
        query = max(set(self.trueLockedFeatures) - set(self.knownLockedCons), key=lambda con: numOfSetsContainFeat(con, self.domPiFeats))
        self.queries = [query]

  def findQuery(self):
    # recompute approximate queries when we do early stopping
    if config.earlyStop is not None: self.computeApproximateQueries()

    if len(self.queries) == 0: return self.answer
    else: return self.queries.pop()

