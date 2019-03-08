import pprint

from lp import lpDualGurobi, computeValue, lpDualCPLEX
from util import powerset
import copy
import numpy
import itertools
import config
from itertools import combinations
from setcover import coverFeat, removeFeat, leastNumElemSets, elementExists,\
  oshimai, killSupersets
from operator import mul

# the querying return these consts that represent safe policies exist / not exist
EXIST = 'exist'
NOTEXIST = 'notexist'

class ConsQueryAgent():
  """
  Find queries in constraint-uncertain mdps.
  """
  def __init__(self, mdp, consStates, consProbs=None, constrainHuman=False):
    """
    can't think of a class it should inherit..

    mdp: a factored mdp
    consSets: [[states that \phi is changed] for \phi in unknown features]
    goalConsStates: states where the goals are not satisfied
    """
    self.mdp = mdp

    # indices of constraints
    self.consStates = consStates
    self.consIndices = range(len(consStates))
    
    self.consProbs = consProbs
    
    # derive different definition of MR
    self.constrainHuman = constrainHuman

    self.allCons = self.consIndices
    
    # used for iterative queries
    self.knownLockedCons = []
    self.knownFreeCons = []
  
  def initialSafePolicyExists(self):
    """
    Run the LP solver with all constraints and see if the LP problem is feasible.
    """
    return self.findConstrainedOptPi(self.allCons)['feasible']

  def findConstrainedOptPi(self, activeCons):
    mdp = copy.copy(self.mdp)

    zeroConstraints = self.constructConstraints(activeCons, mdp)

    if config.METHOD == 'gurobi':
      return lpDualGurobi(mdp, zeroConstraints=zeroConstraints)
    elif config.METHOD == 'cplex':
      return lpDualCPLEX(mdp, zeroConstraints=zeroConstraints)
    elif config.METHOD == 'mcts':
      return MCTS(**mdp)
    else:
      raise Exception('unknown method')

  def updateFeats(self, newFreeCon=None, newLockedCon=None):
    # function as an interface. does nothing by default
    if newFreeCon != None:
      self.knownFreeCons.append(newFreeCon)
    if newLockedCon != None:
      self.knownLockedCons.append(newLockedCon)

  """
  Methods for safe policy improvement
  """
  #FIXME what is this for?? just to check the computation time?
  def findRelevantFeaturesBruteForce(self):
    allConsPowerset = set(powerset(self.allCons))

    for subsetsToConsider in allConsPowerset:
      self.findConstrainedOptPi(subsetsToConsider)

  def findRelevantFeaturesAndDomPis(self):
    """
    Incrementally add dominating policies to a set
    DomPolicies algorithm in the IJCAI paper
    """
    beta = [] # rules to keep
    dominatingPolicies = {}

    allCons = set()
    allConsPowerset = set(powerset(allCons))
    subsetsConsidered = []

    # iterate until no more dominating policies are found
    while True:
      subsetsToConsider = allConsPowerset.difference(subsetsConsidered)

      if len(subsetsToConsider) == 0: break

      # find the subset with the smallest size
      activeCons = min(subsetsToConsider, key=lambda _: len(_))
      #if config.DEBUG: print 'activeCons', activeCons
      subsetsConsidered.append(activeCons)

      skipThisCons = False
      for enf, relax in beta:
        if enf.issubset(activeCons) and len(relax.intersection(activeCons)) == 0:
          # this subset can be ignored
          skipThisCons = True
          break
      if skipThisCons:
        continue

      sol = self.findConstrainedOptPi(activeCons)
      if sol['feasible']:
        x = sol['pi']
        if config.DEBUG:
          printOccSA(x)
          print self.computeValue(x)

        dominatingPolicies[activeCons] = x

        # check violated constraints
        violatedCons = self.findViolatedConstraints(x)

        if config.DEBUG: print 'x violates', violatedCons
      else:
        # infeasible
        violatedCons = ()
        
        if config.DEBUG: print 'infeasible'

      # beta records that we would not enforce activeCons and relax occupiedFeats in the future
      beta.append((set(activeCons), set(violatedCons)))

      allCons.update(violatedCons)

      allConsPowerset = set(powerset(allCons))
      
    domPis = []
    for pi in dominatingPolicies.values():
      if pi not in domPis: domPis.append(pi)
      
    if config.DEBUG: print 'rel cons', allCons, 'num of domPis', len(domPis)
    return allCons, domPis

  def findMinimaxRegretConstraintQBruteForce(self, k, relFeats, domPis):
    mrs = {}

    if len(relFeats) < k:
      # we have a chance to ask about all of them!
      return tuple(relFeats)
    else:
      for q in itertools.combinations(relFeats, k):
        mr, advPi = self.findMRAdvPi(q, relFeats, domPis, k)
        mrs[q] = mr
        
        print q, 'mr', mr

      if mrs == {}:
        mmq = () # no need to ask anything
      else:
        mmq = min(mrs.keys(), key=lambda _: mrs[_])
    
      return mmq

  # REFACTOR these should be in different classes. All implement a findQuery interface
  def findMinimaxRegretConstraintQ(self, k, relFeats, domPis, scopeHeu=True, filterHeu=True):
    """
    Finding a minimax k-element constraint query.
    
    The pruning rule depends on two heuristics: sufficient features (scopeHeu) and query dominance (filterHeu).
    Set each to be true to enable the filtering aspect.
    (We only compared enabling both, which is MMRQ-k, with some baseliens. We didn't analyze the effect of enabling only one of them.)
    """
    if len(relFeats) < k:
      # we have a chance to ask about all of them!
      return tuple(relFeats)

    # candidate queries and their violated constraints
    candQVCs = {}
    mrs = {}

    # first query to consider
    q = self.findChainedAdvConstraintQ(k, relFeats, domPis)
    
    if len(q) < k: return q # mr = 0 in this case

    # all sufficient features
    allCons = set()
    allCons.update(q)

    if scopeHeu:
      allConsPowerset = set(itertools.combinations(allCons, k))
    else:
      allConsPowerset = set(itertools.combinations(relFeats, k))

    qChecked = []

    while True:
      qToConsider = allConsPowerset.difference(qChecked)

      if len(qToConsider) == 0: break

      # find the subset with the smallest size
      q = qToConsider.pop()
      qChecked.append(q)

      # check the pruning condition
      dominatedQ = False
      if filterHeu:
        for candQ in candQVCs.keys():
          if set(q).intersection(candQVCs[candQ]).issubset(candQ):
            dominatedQ = True
            break
        if dominatedQ:
          print q, 'is dominated'
          continue

      mr, advPi = self.findMRAdvPi(q, relFeats, domPis, k)
      
      print q, mr
      #printOccSA(advPi) # for debug

      candQVCs[q] = self.findViolatedConstraints(advPi)
      allCons.update(candQVCs[q])
      if scopeHeu:
        allConsPowerset = set(itertools.combinations(allCons, k))
      # allConsPowerset is consistent (all k-subsets) if not scope. no need to update. 

      mrs[q] = mr
      
    mmq = min(mrs.keys(), key=lambda _: mrs[_])
  
    return mmq

  def findChainedAdvConstraintQ(self, k, relFeats, domPis, informed=False):
    q = []
    while len(q) < k:
      sizeOfQ = len(q)
      
      if informed:
        mr, advPi = self.findMRAdvPi(q, relFeats, domPis, k - sizeOfQ, consHuman=True, tolerated=q)
      else:
        mr, advPi = self.findMRAdvPi(q, relFeats, domPis, k, consHuman=False)
        
      violatedCons = self.findViolatedConstraints(advPi)
      print 'vio cons', violatedCons

      # we want to be careful about this, add unseen features to q
      # not disturbing the order of features in q
      for con in violatedCons:
        if con not in q:
          q.append(con)
      
      if len(q) == sizeOfQ: break # no more constraints to add
    
    # may exceed k constraints. return the first k constraints only
    mmq = list(q)[:k]
    return mmq

  def findRelevantRandomConstraintQ(self, k, relFeats):
    if len(relFeats) == 0: # possibly k == 0, make a separate case here
      return []
    elif len(relFeats) > k:
      relFeats = list(relFeats)
      indices = range(len(relFeats))
      randIndices = numpy.random.choice(indices, k, replace=False)
      q = [relFeats[_] for _ in randIndices]
    else:
      # no more than k relevant features
      q = relFeats
    
    return q
 
  def findRandomConstraintQ(self, k):
    if len(self.consIndices) >= k:
      q = numpy.random.choice(self.consIndices, k, replace=False)
    else:
      # no more than k constraints, should not design exp in this way though
      q = self.consIndices
    
    return q
  
  def findRegret(self, q, violableCons):
    """
    A utility function that finds regret given the true violable constraints
    """
    consRobotCanViolate = set(q).intersection(violableCons)
    rInvarCons = set(self.allCons).difference(consRobotCanViolate)
    robotPi = self.findConstrainedOptPi(rInvarCons)['pi']
    
    hInvarCons = set(self.allCons).difference(violableCons)
    humanPi = self.findConstrainedOptPi(hInvarCons)['pi']
    
    hValue = self.computeValue(humanPi)
    rValue = self.computeValue(robotPi)
    
    regret = hValue - rValue
    assert regret >= -0.00001, 'human %f, robot %f' % (hValue, rValue)

    return regret

  def findRobotDomPis(self, q, relFeats, domPis):
    """
    Find the set of dominating policies adoptable by the robot.
    """
    invarFeats = set(relFeats).difference(q)
    pis = []

    for rPi in domPis:
      if self.piSatisfiesCons(rPi, invarFeats):
        pis.append(rPi)

    return pis

  def findMRAdvPi(self, q, relFeats, domPis, k, consHuman=None, tolerated=[]):
    """
    Find the adversarial policy given q and domPis
    
    consHuman can be set to override self.constrainHuman
    makesure that |humanViolated \ tolerated| <= k
    
    Now searching over all dominating policies, maybe take some time.. can use MILP instead?
    """
    if consHuman == None: consHuman = self.constrainHuman

    maxRegret = 0
    advPi = None

    for pi in domPis:
      humanViolated = self.findViolatedConstraints(pi)
      humanValue = self.computeValue(pi)

      if consHuman and len(set(humanViolated).difference(tolerated)) > k:
        # we do not consider the case where the human's optimal policy violates more than k constraints
        # unfair to compare.
        continue

      # intersection of q and constraints violated by pi
      consRobotCanViolate = set(q).intersection(humanViolated)
      
      # the robot's optimal policy given the constraints above
      invarFeats = set(relFeats).difference(consRobotCanViolate)
      
      robotValue = -numpy.inf
      robotPi = None
      for rPi in domPis:
        if self.piSatisfiesCons(rPi, invarFeats):
          rValue = self.computeValue(rPi)
          if rValue > robotValue:
            robotValue = rValue
            robotPi = rPi

      regret = humanValue - robotValue
      
      assert robotPi != None
      # ignore numerical issues
      assert regret >= -0.00001, 'human %f, robot %f' % (humanValue, robotValue)

      if regret > maxRegret or (regret == maxRegret and advPi == None):
        maxRegret = regret
        advPi = pi
  
    # even with constrainHuman, the non-constraint-violating policy is in \Gamma
    assert advPi != None
    return maxRegret, advPi

  def constructConstraints(self, cons, mdp):
    """
    The set of state, action pairs that should not be visited when cons are active constraints.
    """
    return [(s, a) for a in mdp.A for con in cons for s in self.consStates[con]]

  def computeValue(self, x):
    return computeValue(x, self.mdp.r, self.mdp.S, self.mdp.A)

  def piSatisfiesCons(self, x, cons):
    violatedCons = self.findViolatedConstraints(x)
    return set(cons).isdisjoint(set(violatedCons))

  def findViolatedConstraints(self, x):
    # set of changed features
    var = set()

    for idx in self.consIndices:
      # states violated by idx
      for s, a in x.keys():
        if any(x[s, a] > 0 for a in self.mdp.A) and s in self.consStates[idx]:
          var.add(idx)
    
    return set(var)

  """
  Methods for finding sets useful for safe policies.
  """
  def safePolicyExist(self, freeCons=None):
    # some dom pi's relevant features are all free
    if freeCons == None:
      freeCons = self.knownFreeCons

    if hasattr(self, 'piRelFeats'):
      # if we have rel featus, simply check whether we covered all rel feats of any dom pi
      return any(len(set(relFeats) - set(freeCons)) == 0 for relFeats in self.piRelFeats)
    else:
      # for some simple heuristics, it's not fair to ask them to precompute dompis (need to run a lot of LP)
      # so we try to solve the lp problem once here
      # see whether the lp is feasible if we assume all other features are locked
      return self.findConstrainedOptPi(set(self.allCons) - set(freeCons))['feasible']
  
  def safePolicyNotExist(self, lockedCons=None):
    # there are some locked features in all dom pis
    if lockedCons == None:
      lockedCons = self.knownLockedCons
    if hasattr(self, 'piRelFeats'):
      return all(len(set(relFeats).intersection(lockedCons)) > 0 for relFeats in self.piRelFeats)
    else:
      # by only imposing these constraints, see whether the lp problem is infeasible
      return not self.findConstrainedOptPi(lockedCons)['feasible']
 
  def checkSafePolicyExists(self):
    """
    None if can't claim, otherwise return exists or notExist
    """
    if self.safePolicyExist(): return EXIST
    elif self.safePolicyNotExist(): return NOTEXIST
    else: return None

  def computePolicyRelFeats(self):
    """
    Compute relevant features of dominating policies.
    If the relevant features of any dominating policy are all free, then safe policies exist.
    Put in another way, if all dom pis has at least one locked relevant feature, then safe policies do not exist.
    
    This can be O(2^|relevant features|), depending on the implementation of findDomPis
    """
    # check whether this is already computed
    if hasattr(self, 'piRelFeats'): return

    relFeats, domPis = self.findRelevantFeaturesAndDomPis()
    piRelFeats = []
    
    for domPi in domPis:
      piRelFeats.append(tuple(self.findViolatedConstraints(domPi)))
    
    # just to remove supersets
    piRelFeats = killSupersets(piRelFeats)

    self.piRelFeats = piRelFeats
    self.relFeats = relFeats # the union of rel feats of all dom pis

  def computeIISs(self):
    """
    Compute IISs by looking at relevant features of dominating policies.

    eg. (1 and 2) or (3 and 4) --> (1 or 3) and (1 or 4) and (2 or 3) and (2 or 4) 
    """
    # we first need relevant features
    if not hasattr(self, 'piRelFeats'):
      self.computePolicyRelFeats()
      
    iiss = [set()]
    for relFeats in self.piRelFeats:
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
    unknownConsPowerset = powerset(self.allCons)
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

  def statesWithDifferentFeats(self, idx, mdp):
    return filter(lambda s: s[idx] != mdp.s0[idx], mdp.S)

  # FIXME remove or not? only used by depreciated methods
  """
  def statesTransitToDifferentFeatures(self, idx, value):
    ret = []
    for s in self.mdp['S']:
      if s[idx] == value:
        for a in self.mdp['A']:
          for sp in self.mdp['S']:
            if self.mdp['T'](s, a, sp) > 0 and sp[idx] != value:
              ret.append((s, a))
              break
    return ret
  """


class GreedyForSafetyAgent(ConsQueryAgent):
  def __init__(self, mdp, consStates, consProbs=None, constrainHuman=False, useIIS=True, useRelPi=True):
    ConsQueryAgent.__init__(self, mdp, consStates, consProbs, constrainHuman)
    
    self.useIIS = useIIS
    self.useRelPi = useRelPi

    # find all IISs without knowing any locked or free cons
    if self.useRelPi:
      self.computePolicyRelFeats()
    if self.useIIS:
      self.computeIISs()

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
    expNumRemaingSets = {}
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

      expNumRemaingSets[con] = self.consProbs[con] * numWhenFree + (1 - self.consProbs[con]) * numWhenLocked
      
    return min(expNumRemaingSets.iteritems(), key=lambda _: _[1])[0]
  
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
      readyToEvalSet += filter(lambda (l, f): self.getQueryAndValue(l, f) == None and readyToEvaluate(l, f),\
                               [(set(lockedCons) - {cons}, freeCons) for cons in lockedCons] +\
                               [(lockedCons, set(freeCons) - {cons}) for cons in freeCons])

  def findQuery(self):
    # we only care about the categories of rel feats
    relLockedCons = set(self.knownLockedCons).intersection(self.relFeats)
    relFreeCons = set(self.knownFreeCons).intersection(self.relFeats)

    qAndV = self.getQueryAndValue(relLockedCons, relFreeCons)
    assert qAndV != None
    return qAndV[0]
 

def printOccSA(x):
  nonZeroSAOcc = filter(lambda _: _[1] > 0, x.items())
  pprint.pprint(sorted(nonZeroSAOcc, key=lambda _: _[0][0][-1]))
