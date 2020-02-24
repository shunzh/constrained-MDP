import time
from operator import mul

from lp import lpDualGurobi, computeValue, lpDualCPLEX
from util import powerset, printOccSA
import config

# the querying return these consts that represent safe policies exist / not exist
EXIST = 'exist'
NOTEXIST = 'notexist'

class ConsQueryAgent():
  """
  Find queries in constraint-uncertain mdps.
  """
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None, knownLockedCons=(), knownFreeCons=()):
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
    self.adversarial = (consProbs is None)

    self.goalCons = [(s, a) for a in mdp.A for s in goalStates]

    self.knownLockedCons = list(knownLockedCons)
    self.knownFreeCons = list(knownFreeCons)
    self.unknownCons = list(set(self.consIndices) - set(self.knownLockedCons) - set(self.knownFreeCons))

  def initialSafePolicyExists(self):
    """
    Run the LP solver with all constraints and see if the LP problem is feasible.
    """
    statusObj = self.findConstrainedOptPi(self.unknownCons)

    return statusObj['feasible']

  def updateFeats(self, newFreeCon=None, newLockedCon=None):
    """
    update feature partition after response.
    #fixme this should be only useful for sequential query agent
    """
    if newFreeCon is not None:
      self.unknownCons.remove(newFreeCon)
      self.knownFreeCons.append(newFreeCon)
    if newLockedCon is not None:
      self.unknownCons.remove(newLockedCon)
      self.knownLockedCons.append(newLockedCon)

  def getLockedFeatCons(self):
    return [self.consStates[idx] for idx in self.knownLockedCons]

  def getUnknownFeatCons(self):
    return [self.consStates[idx] for idx in self.unknownCons]

  def getGivenFeatCons(self, indices):
    return [self.consStates[idx] for idx in indices]

  def findConstrainedOptPi(self, activeCons=(), addKnownLockedCons=True, mdp=None):
    """
    :param activeCons:  constraints that should be followed
    :param mdp: use mdp.r by default
    :return: {'feasible': if solution exists; if not exists, this is the only property,
              'obj': the objective value,
              'pi': the policy found}
    """
    if mdp is None: mdp = self.mdp

    if addKnownLockedCons:
      activeCons = tuple(activeCons) + tuple(self.knownLockedCons)
    zeroConstraints = self.getGivenFeatCons(activeCons)

    if config.OPT_METHOD == 'gurobi':
      return lpDualGurobi(mdp, zeroConstraints=zeroConstraints, positiveConstraints=self.goalCons)
    elif config.OPT_METHOD == 'cplex':
      # not using this. only for comparision
      return lpDualCPLEX(mdp, zeroConstraints=zeroConstraints, positiveConstraints=self.goalCons)
    else:
      raise Exception('unknown method')

  """
  Useful wrapper functions for the one above.
  """
  def computeCurrentSafelyOptPi(self):
    return self.findConstrainedOptPi(activeCons=self.unknownCons)['pi']

  def computeCurrentSafelyOptPiValue(self):
    return self.findConstrainedOptPi(activeCons=self.unknownCons)['obj']

  """
  Methods for finding dominating policies and relevant features
  """
  def findRelevantFeaturesBruteForce(self):
    """
    a method simply to measure the time needed to compute all dominating policies
    """
    allConsPowerset = set(powerset(self.unknownCons))

    for subsetsToConsider in allConsPowerset:
      self.findConstrainedOptPi(subsetsToConsider)

  def findRelevantFeaturesAndDomPis(self):
    """
    Incrementally add dominating policies to a set
    DomPolicies algorithm in the IJCAI paper

    earlyStop: stop within this time and return whatever dompis found
    """
    beta = [] # rules to keep
    dominatingPolicies = {}
    dominatingPiValues = {}

    allCons = set()
    allConsPowerset = set(powerset(allCons))
    subsetsConsidered = []

    if config.earlyStop is None:
      # never stop before finding all dom pis
      terminateCond = lambda: False
    else:
      startTime = time.time()
      terminateCond = lambda: time.time() - startTime >= config.earlyStop

    # iterate until no more dominating policies are found
    while not terminateCond():
      subsetsToConsider = allConsPowerset.difference(subsetsConsidered)

      if len(subsetsToConsider) == 0: break

      # find the subset with the smallest size
      activeCons = min(subsetsToConsider, key=lambda _: len(_))
      if config.DEBUG: print 'activeCons', activeCons
      subsetsConsidered.append(activeCons)

      skipThisCons = False
      for enf, relax in beta:
        if enf.issubset(activeCons) and len(relax.intersection(activeCons)) == 0:
          # this subset can be ignored
          skipThisCons = True
          if config.DEBUG: print 'dominated'
          break
      if skipThisCons:
        continue

      # it will enforce activeCons and known locked features (inside)
      sol = self.findConstrainedOptPi(list(activeCons))
      if sol['feasible']:
        x = sol['pi']
        # check violated constraints
        violatedCons = self.findViolatedConstraints(x)

        # if an old dominating policy violates more constraints than the current one, but not have as high value as the current one
        # then remove that old dominating policy
        for oldCons, oldViolated in beta:
          oldCons = tuple(oldCons)
          if oldCons in dominatingPiValues.keys() and set(violatedCons).issubset(oldViolated) and dominatingPiValues[oldCons] <= sol['obj']:
            # the dom pi under cons is dominated by the current pi
            dominatingPolicies.pop(oldCons)
            dominatingPiValues.pop(oldCons)

        dominatingPolicies[activeCons] = x
        dominatingPiValues[activeCons] = sol['obj']

        if config.DEBUG: print 'this policy violates', violatedCons
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

    # make sure returned values are lists
    allCons = list(allCons)
    if config.DEBUG: print 'rel cons', allCons, 'num of domPis', len(domPis)
    return allCons, domPis

  def computeValue(self, x):
    """
    compute the value of policy x. it computes the dot product between x and r
    """
    return computeValue(x, self.mdp.r, self.mdp.S, self.mdp.A)

  def piSatisfiesCons(self, x, cons):
    violatedCons = self.findViolatedConstraints(x)
    return set(cons).isdisjoint(set(violatedCons))

  def findViolatedConstraints(self, x):
    """
    only return the indices of unknown features that are changed by policy (w/ occupancy x)
    """
    var = []

    for idx in self.unknownCons:
      # states violated by idx
      for s, a in x.keys():
        if any(x[s, a] > 0 for a in self.mdp.A) and s in self.consStates[idx]:
          var.append(idx)
          break
    
    return var

  # syntax sugar functions for computing \prod_{feat} p_f(feat)
  def probFeatsBeingFree(self, feats):
    return reduce(mul, map(lambda _: self.consProbs[_], feats), 1)
  def probFeatsBeingLocked(self, feats):
    return reduce(mul, map(lambda _: 1 - self.consProbs[_], feats), 1)
