import pprint

from lp import lpDualGurobi, computeValue, lpDualCPLEX
from util import powerset
import copy
import numpy
import config
from itertools import combinations
from setcover import killSupersets

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

  def findMRAdvPi(self, q, relFeats, domPis, k, consHuman=None, tolerated=[]):
    """
    Find the adversarial policy given q and domPis
    
    consHuman can be set to override self.constrainHuman
    make sure that |humanViolated \ tolerated| <= k
    
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
    """
    compute the value of policy x. it computes the dot product between x and r
    """
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


def printOccSA(x):
  nonZeroSAOcc = filter(lambda _: _[1] > 0, x.items())
  pprint.pprint(sorted(nonZeroSAOcc, key=lambda _: _[0][0][-1]))
