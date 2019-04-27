import numpy

from algorithms.consQueryAgents import ConsQueryAgent
import itertools

class SafeImproveAgent(ConsQueryAgent):
  """
  functions for finding better policies than an initial safe policy

  FIXME inconsistency. algorithms are implemented as methods here but as classes in initialSafeAgent
  """
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

  def findMRAdvPi(self, q, relFeats, domPis, k, consHuman=None, tolerated=()):
    """
    Find the adversarial policy given q and domPis

    consHuman can be set to override self.constrainHuman
    make sure that |humanViolated \ tolerated| <= k

    Now searching over all dominating policies, maybe take some time.. can use MILP instead?
    """
    if consHuman is None: consHuman = self.constrainHuman

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

      assert robotPi is not None
      # ignore numerical issues
      assert regret >= -0.00001, 'human %f, robot %f' % (humanValue, robotValue)

      if regret > maxRegret or (regret == maxRegret and advPi is None):
        maxRegret = regret
        advPi = pi

    # even with constrainHuman, the non-constraint-violating policy is in \Gamma
    assert advPi is not None
    return maxRegret, advPi

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

