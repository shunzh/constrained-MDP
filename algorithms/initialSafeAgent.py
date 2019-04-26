import copy
import random

import config
from algorithms import lp
from algorithms.consQueryAgents import ConsQueryAgent, NOTEXIST, EXIST
from algorithms.setcover import coverFeat, removeFeat, leastNumElemSets
from operator import mul

from util import powerset


class GreedyForSafetyAgent(ConsQueryAgent):
  def __init__(self, mdp, consStates, consProbs=None, constrainHuman=False, useIIS=True, useRelPi=True, adversarial=False, optimizeValue=False):
    ConsQueryAgent.__init__(self, mdp, consStates, consProbs, constrainHuman)

    self.useIIS = useIIS
    self.useRelPi = useRelPi

    # assume the human's response is adversarial, used in the non-Bayesian case
    self.adversarial = adversarial
    # set if the robot should also aim for finding a safe policy with higher value
    self.optimizeValue = optimizeValue

    # find all IISs without knowing any locked or free cons
    if self.useRelPi:
      self.computePolicyRelFeats()

    if self.useIIS:
      self.computeIISs()

    if self.optimizeValue:
      self.computeFeatureValue()

  def computeFeatureValue(self):
    """
    Compute how much each feature worth for finding a high-valued safe policy.
    saves to self.featureVals
    """
    # otherwise cannot compute rel feat values
    assert self.useRelPi

    n = len(self.piRelFeats)
    d = len(self.relFeats)

    A = [[1 if self.relFeats[j] in self.piRelFeats[i] else 0 for j in xrange(d)] for i in xrange(n)]
    b = [self.piRelFeatsAndValues[tuple(self.piRelFeats[i])] for i in xrange(n)]
    print A, b
    weights = lp.linearRegression(A, b)

    self.featureVals = {}
    for i in range(d): self.featureVals[self.relFeats[i]] = weights[i]
    print 'feat vals', self.featureVals

  def updateFeats(self, newFreeCon=None, newLockedCon=None):
    # this just add to the list of known free and locked features
    ConsQueryAgent.updateFeats(self, newFreeCon, newLockedCon)

    if newFreeCon != None:
      if self.useIIS:
        self.iiss = coverFeat(newFreeCon, self.iiss)
      if self.useRelPi:
        self.piRelFeats = removeFeat(newFreeCon, self.piRelFeats)

    if newLockedCon != None:
      if self.useIIS:
        self.iiss = removeFeat(newLockedCon, self.iiss)
      if self.useRelPi:
        self.piRelFeats = coverFeat(newLockedCon, self.piRelFeats)

  def findQuery(self):
    """
    return the next feature to query by greedily cover the most number of sets
    return None if no more features are needed or nothing left to query about
    """
    answerFound = self.checkSafePolicyExists()
    if answerFound != None: return answerFound

    # make sure the constraints that are already queried are not going to be queried again
    unknownCons = set(self.consIndices) - set(self.knownFreeCons) - set(self.knownLockedCons)

    # find the maximum frequency constraint weighted by the probability
    score = {}
    for con in unknownCons:
      # prefer using iis
      if self.useIIS and self.useRelPi:
        numWhenFree = len(coverFeat(con, self.iiss))
        numWhenLocked = len(coverFeat(con, self.piRelFeats))
      elif self.useIIS and (not self.useRelPi):
        numWhenFree = len(coverFeat(con, self.iiss))
        numWhenLocked = len(leastNumElemSets(con, self.iiss))
      elif (not self.useIIS) and self.useRelPi:
        numWhenFree = len(leastNumElemSets(con, self.piRelFeats))
        numWhenLocked = len(coverFeat(con, self.piRelFeats))
      else:
        # not useRelPi and not useIIS, can't be the case
        raise Exception('no idea what to do in this case')

      if self.adversarial:
        # non-Bayesian
        if not self.optimizeValue:
          # not dependent on consProbs
          score[con] = max(numWhenFree, numWhenLocked)
        else:
          assert self.useIIS
          score[con] = self.featureVals[con] / len(filter(lambda _: con in _, self.iiss))
      else:
        score[con] = self.consProbs[con] * numWhenFree + (1 - self.consProbs[con]) * numWhenLocked

    return min(score.iteritems(), key=lambda _: _[1])[0]

  def heuristic(self, con):
    raise Exception('need to be defined')


class DomPiHeuForSafetyAgent(ConsQueryAgent):
  """
  This uses dominating policies. It first finds the dominating policy that has the largest probability being free.
  Then query about the most probable unknown feature in the relevant features of the policy.
  """
  def __init__(self, mdp, consStates, consProbs=None, constrainHuman=False):
    ConsQueryAgent.__init__(self, mdp, consStates, consProbs, constrainHuman)

    self.computePolicyRelFeats()

  def findQuery(self):
    answerFound = self.checkSafePolicyExists()
    if answerFound != None: return answerFound

    unknownCons = set(self.consIndices) - set(self.knownFreeCons) - set(self.knownLockedCons)

    # find the most probable policy
    # update the cons prob to make it easier
    updatedConsProbs = copy.copy(self.consProbs)
    for i in self.consIndices:
      if i in self.knownLockedCons: updatedConsProbs[i] = 0
      elif i in self.knownFreeCons: updatedConsProbs[i] = 1

    # find the policy that has the largest probability to be feasible

    feasibleProb = lambda relFeats: reduce(mul,\
                                           map(lambda _: updatedConsProbs[_], relFeats),\
                                           1)

    maxProbPiRelFeats = max(self.piRelFeats, key=feasibleProb)

    # now query about unknown features in the most probable policy's relevant features
    featsToConsider = unknownCons.intersection(maxProbPiRelFeats)
    # the probability is less than 1. so there must be unknown features to consider
    assert len(featsToConsider) > 0
    return max(featsToConsider, key=lambda _: self.consProbs[_])


class MaxProbSafePolicyExistAgent(ConsQueryAgent):
  """
  Find the feature that, after querying, the expected probability of finding a safe poicy / no safe policies exist is maximized.
  """
  def __init__(self, mdp, consStates, consProbs=None, constrainHuman=False):
    ConsQueryAgent.__init__(self, mdp, consStates, consProbs, constrainHuman)

    # need domPis for query
    self.computePolicyRelFeats()

  def probOfExistanceOfSafePolicies(self, lockedCons, freeCons):
    """
    Compute the probability of existence of at least one safe policies.
    This considers the changeabilities of all unknown features.

    lockedCons, freeCons: The set of locked and free features.
      They might be different from the ones confirmed by querying.
      These are hypothetical ones just to compute the corresponding prob.
    """
    unknownCons = set(self.consIndices) - set(lockedCons) - set(freeCons)
    # \EE[policy exists]
    expect = 0

    allSubsetsOfUnknownCons = powerset(unknownCons)

    for freeSubset in allSubsetsOfUnknownCons:
      # assume now freeSubset is free and uknownCons \ freeSubset is locked
      # compute the prob. that this happens (given lockedCons and freeCons)
      prob = reduce(mul, map(lambda _: self.consProbs[_], freeSubset), 1) *\
             reduce(mul, map(lambda _: 1 - self.consProbs[_], unknownCons - set(freeSubset)), 1)

      # an indicator represents if safe policies exist
      safePolicyExist = self.safePolicyExist(freeCons=list(freeCons) + list(freeSubset))

      expect += safePolicyExist * prob

    return expect

  def findQuery(self):
    answerFound = self.checkSafePolicyExists()
    if answerFound != None: return answerFound

    unknownCons = set(self.consIndices) - set(self.knownLockedCons) - set(self.knownFreeCons)

    # the probability that either
    termProbs = {}
    for con in unknownCons:
      # the prob that safe policies exist when con is free
      probExistWhenFree = self.probOfExistanceOfSafePolicies(self.knownLockedCons, self.knownFreeCons + [con])

      termProbs[con] = self.consProbs[con] * probExistWhenFree

    # there should be unqueried features
    assert len(termProbs) > 0

    return max(termProbs.iteritems(), key=lambda _: _[1])[0]


class DescendProbQueryForSafetyAgent(ConsQueryAgent):
  """
  Return the unknown feature that has the largest (or smallest) probability of being changeable.
  """
  def findQuery(self):
    answerFound = self.checkSafePolicyExists()
    if answerFound != None: return answerFound

    unknownCons = set(self.consIndices) - set(self.knownLockedCons) - set(self.knownFreeCons)
    return max(unknownCons, key=lambda con: self.consProbs[con])


class RandomQueryAgent(ConsQueryAgent):
  def findQuery(self):
    answerFound = self.checkSafePolicyExists()
    if answerFound != None: return answerFound

    unknownCons = set(self.consIndices) - set(self.knownLockedCons) - set(self.knownFreeCons)
    return random.choice(unknownCons)


class OptQueryForSafetyAgent(ConsQueryAgent):
  """
  Find the opt query by dynamic programming. Its O(2^|\Phi|).
  """
  def __init__(self, mdp, consStates, consProbs=None, constrainHuman=False):
    ConsQueryAgent.__init__(self, mdp, consStates, consProbs, constrainHuman)

    # need domPis for query
    self.computePolicyRelFeats()

    self.optQs = {}
    # the set of cons imposing which we have a safe policy
    self.freeBoundary = []
    # the set of cons imposing which we don't have a safe policy for sure
    self.lockedBoundary = []

    self.computeOptQueries()

  def getQueryAndValue(self, locked, free):
    if any(set(locked).issuperset(lockedB) for lockedB in self.lockedBoundary):
      return (NOTEXIST, 0)
    elif any(set(free).issuperset(freeB) for freeB in self.freeBoundary):
      return (EXIST, 0)
    elif not (frozenset(locked), frozenset(free)) in self.optQs:
      return None
    else:
      return self.optQs[frozenset(locked), frozenset(free)]

  def setQueryAndValue(self, locked, free, qAndV):
    self.optQs[frozenset(locked), frozenset(free)] = qAndV

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
    admissibleFreeCons = []
    admissibleLockedCons = []
    # the set of (lockedCons, freeCons) to evaluate the optimal queries
    # it's the cross product of the two sets above, excluding free and locked cons that share elements
    admissibleCons = []

    for lockedCons in consPowerset:
      if self.safePolicyNotExist(lockedCons=lockedCons):
        if not any(set(lockedCons).issuperset(lockedB) for lockedB in self.lockedBoundary):
          self.lockedBoundary.append(lockedCons)
      else: admissibleLockedCons.append(lockedCons)

    for freeCons in consPowerset:
      if self.safePolicyExist(freeCons=freeCons):
        if not any(set(freeCons).issuperset(freeB) for freeB in self.freeBoundary):
          self.freeBoundary.append(freeCons)
      else: admissibleFreeCons.append(freeCons)

    if config.VERBOSE:
      print 'locked', self.lockedBoundary
      print 'free', self.freeBoundary

    for lockedCons in admissibleLockedCons:
      for freeCons in admissibleFreeCons:
        # any cons should not be known to be both free and locked
        if set(lockedCons).isdisjoint(set(freeCons)):
          admissibleCons.append((lockedCons, freeCons))

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
      if config.VERBOSE: print len(readyToEvalSet), 'need to be evaluated'

      (lockedCons, freeCons) = readyToEvalSet.pop()

      unknownCons = set(self.relFeats) - set(lockedCons) - set(freeCons)

      minNums = [(con,\
                  self.consProbs[con] * self.getQueryAndValue(lockedCons, set(freeCons).union({con}))[1]\
                  + (1 - self.consProbs[con]) * self.getQueryAndValue(set(lockedCons).union({con}), freeCons)[1]\
                  + 1) # count con in
                 for con in unknownCons]
      # pick the tuple that has the minimum obj value after querying
      self.setQueryAndValue(lockedCons, freeCons, min(minNums, key=lambda _: _[1]))

      # add neighbors that ready to evaluate to readToEvalSet
      readyToEvalSet += filter(lambda (l, f): self.getQueryAndValue(l, f) == None and readyToEvaluate(l, f),
                               [(set(lockedCons) - {cons}, freeCons) for cons in lockedCons] +\
                               [(lockedCons, set(freeCons) - {cons}) for cons in freeCons])

  def findQuery(self):
    # we only care about the categories of rel feats
    relLockedCons = set(self.knownLockedCons).intersection(self.relFeats)
    relFreeCons = set(self.knownFreeCons).intersection(self.relFeats)

    qAndV = self.getQueryAndValue(relLockedCons, relFreeCons)
    assert qAndV != None
    return qAndV[0]
