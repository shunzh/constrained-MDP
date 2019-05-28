import pprint

from lp import lpDualGurobi, computeValue, lpDualCPLEX
from util import powerset
import copy
import config

# the querying return these consts that represent safe policies exist / not exist
EXIST = 'exist'
NOTEXIST = 'notexist'

class ConsQueryAgent():
  """
  Find queries in constraint-uncertain mdps.
  """
  def __init__(self, mdp, consStates, goalStates=(), consProbs=None):
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

    self.allCons = self.consIndices # FIXME different subclasses call this differently!

    self.goalCons = [(s, a) for a in mdp.A for s in goalStates]

    # used for iterative queries
    self.knownLockedCons = []
    self.knownFreeCons = []
  
  def initialSafePolicyExists(self):
    """
    Run the LP solver with all constraints and see if the LP problem is feasible.
    """
    return self.findConstrainedOptPi(self.allCons)['feasible']

  def findConstrainedOptPi(self, activeCons):
    """
    :param activeCons:  constraints that should be followed
    :return: {'feasible': if solution exists; if not exists, this is the only property,
              'obj': the objective value,
              'pi': the policy found}
    """
    mdp = copy.copy(self.mdp)

    zeroConstraints = self.constructConstraints(activeCons)

    if config.METHOD == 'gurobi':
      return lpDualGurobi(mdp, zeroConstraints=zeroConstraints, positiveConstraints=self.goalCons)
    elif config.METHOD == 'cplex':
      # not using this. only for comparision
      return lpDualCPLEX(mdp, zeroConstraints=zeroConstraints, positiveConstraints=self.goalCons)
    else:
      raise Exception('unknown method')

  def updateFeats(self, newFreeCon=None, newLockedCon=None):
    # function as an interface. does nothing by default
    if newFreeCon is not None:
      self.knownFreeCons.append(newFreeCon)
    if newLockedCon is not None:
      self.knownLockedCons.append(newLockedCon)

  """
  Methods for finding dominating policies and relevant features
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

      sol = self.findConstrainedOptPi(activeCons)
      if sol['feasible']:
        x = sol['pi']
        if config.DEBUG:
          printOccSA(x)
          print self.computeValue(x)

        dominatingPolicies[activeCons] = x

        # check violated constraints
        violatedCons = self.findViolatedConstraints(x)

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

  def constructConstraints(self, cons):
    """
    The set of state, action pairs that should not be visited when cons are active constraints.
    """
    mdp = self.mdp
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
    
    return list(var)



def printOccSA(x):
  nonZeroSAOcc = filter(lambda _: _[1] > 0, x.items())
  pprint.pprint(sorted(nonZeroSAOcc, key=lambda _: _[0][0][-1]))
