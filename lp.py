try:
  from gurobipy import *
except ImportError:
  print "can't import gurobipy"

try:
  from pycpx import CPlexModel, CPlexException, CPlexNoSolution
except ImportError:
  print "can't import pycpx"

import easyDomains
import config
import util

def lp(S, A, r, T, s0):
  """
  Solve the LP problem to find out the optimal occupancy
  
  Args:
    S: state set
    A: action set
    r: reward
    T: transition function
    s0: init state
  """
  m = CPlexModel()
  if not config.VERBOSE or config.DEBUG: m.setVerbosity(0)

  # useful constants
  Sr = range(len(S))
 
  v = m.new(len(S), name='v')

  for s in Sr:
    for a in A:
      m.constrain(v[s] >= r(S[s], a) + sum(v[sp] * T(S[s], a, S[sp]) for sp in Sr))
  
  # obj
  obj = m.minimize(v[s0])
  ret = util.Counter()
  for s in Sr:
    ret[S[s]] = m[v][s]
  return ret

def lpDualGurobi(mdp, zeroConstraints=[], positiveConstraints=[], positiveConstraintsOcc=1):
  """
  Solve the dual problem of lp, maybe with some constraints
  Same arguments
  
  Note that this is a lower level function that does not consider feature extraction.
  r should be a reward function, not a reward parameter.
  """
  S = mdp.S
  A = mdp.A
  T = mdp.T
  r = mdp.r
  gamma = mdp.gamma
  alpha = mdp.alpha
  
  # initialize a Gurobi model
  m = Model()
  m.setParam('OutputFlag', False)

  # useful constants
  Sr = range(len(S))
  Ar = range(len(A))

  x = m.addVars(len(S), len(A), lb=0, name='x')

  # flow conservation constraints
  for sp in Sr:
    # x (x(s) - \gamma * T) = \sigma
    m.addConstr(sum(x[s, a] * ((s == sp) - gamma * T(S[s], A[a], S[sp])) for s in Sr for a in Ar) == alpha(S[sp]))
  
  # == constraints
  if len(zeroConstraints) > 0:
    m.addConstr(sum(x[S.index(s), A.index(a)] for s, a in zeroConstraints) == 0)

  # >= constraints. the occupancy should be at least positiveConstraintsOcc
  if len(positiveConstraints) > 0:
    m.addConstr(sum(x[S.index(s), A.index(a)] for s, a in positiveConstraints) >= positiveConstraintsOcc)
    
  # obj
  m.setObjective(sum([x[s, a] * r(S[s], A[a]) for s in Sr for a in Ar]), GRB.MAXIMIZE)

  m.optimize()
  
  if m.status == GRB.Status.OPTIMAL: 
    # return feasible being true and the obj value, opt pi
    # .X attribute is to retrieve the value of the variable
    return {'feasible': True, 'obj': m.objVal, 'pi': {(S[s], A[a]): x[s, a].X for s in Sr for a in Ar}}
  elif m.status == GRB.Status.INF_OR_UNBD:
    # simply return infeasible
    return {'feasible': False}
  else:
    raise Exception('error status: %d' % m.status)


def lpDualCPLEX(mdp, zeroConstraints=[], positiveConstraints=[], positiveConstraintsOcc=1):
  """
  Solve the dual problem of lp, maybe with some constraints
  Same arguments

  Note that this is a lower level function that does not consider feature extraction.
  r should be a reward function, not a reward parameter.
  """

  S = mdp.S
  A = mdp.A
  T = mdp.T
  r = mdp.r
  gamma = mdp.gamma
  alpha = mdp.alpha

  m = CPlexModel()
  if not config.VERBOSE: m.setVerbosity(0)

  # useful constants
  Sr = range(len(S))
  Ar = range(len(A))

  x = m.new((len(S), len(A)), lb=0, name='x')

  # make sure x is a valid occupancy
  for sp in Sr:
    # x (x(s) - \gamma * T) = \sigma
    m.constrain(sum(x[s, a] * ((s == sp) - gamma * T(S[s], A[a], S[sp])) for s in Sr for a in Ar) == alpha(S[sp]))

  # == constraints
  if len(zeroConstraints) > 0:
    m.constrain(sum(x[S.index(s), A.index(a)] for s, a in zeroConstraints) == 0)

  # >= constraints
  if len(positiveConstraints) > 0:
    m.constrain(sum(x[S.index(s), A.index(a)] for s, a in positiveConstraints) >= positiveConstraintsOcc)

  # obj
  try:
    obj = m.maximize(sum([x[s, a] * r(S[s], A[a]) for s in Sr for a in Ar]))
  except CPlexException as err:
    print 'Exception', err
    # we return obj value as None and occ measure as {}. this should be handled correctly
    return {'feasible': False}

  return {'feasible': True, 'obj': obj, 'pi': {(S[s], A[a]): m[x][s, a] for s in Sr for a in Ar}}


def decomposePiLP(S, A, T, s0, terminal, rawX, x, gamma=1):
  """
  DEPRECATED.
  This tries to decouple a policy into the optimal policy (following no constraints) and another policy \pi'.
  \pi' may be a dominating policy.
  Described in Eq. 2 on Aug.29, 2017.
  """
  m = CPlexModel()
  if not config.VERBOSE: m.setVerbosity(0)

  # useful constants
  Sr = range(len(S))
  Ar = range(len(A))
 
  y = m.new((len(S), len(A)), lb=0, name='y')
  sigma = m.new(lb=0, ub=1, name='sigma')
  
  for s in Sr:
    for a in Ar:
      # note that x and rawX use S x A as domains
      m.constrain(sigma * rawX[S[s], A[a]] + y[s, a] == x[S[s], A[a]])

  # make sure y is a valid occupancy
  for sp in Sr:
    # x (x(s) - \gamma * T) = \sigma
    # and make sure there is no flow back from the terminal states
    if not terminal(S[sp]):
      m.constrain(sum(y[s, a] * ((s == sp) - gamma * T(S[s], A[a], S[sp]) * (not terminal(S[s]))) for s in Sr for a in Ar) == (1 - sigma) * (S[sp] == s0))
    else:
      m.constrain(sum(y[s, a] * ((s == sp) - gamma * T(S[s], A[a], S[sp]) * (not terminal(S[sp]))) for s in Sr for a in Ar) == (1 - sigma) * (S[sp] == s0))
 
  obj = m.maximize(sigma)
  
  # return sigma and the value of y
  return obj, {(S[s], A[a]): m[y][s, a] for s in Sr for a in Ar}


def milp(S, A, R, T, s0, psi, maxV):
  """
  Solve the MILP problem in greedy construction of policy query
  
  Args:
    S: state set
    A: action set
    R: reward candidate set
    T: transition function
    s0: init state
    psi: prior belief on rewards
    maxV: maxV[i] = max_{\pi \in q} V_{r_i}^\pi
  """
  m = CPlexModel()
  if not config.VERBOSE: m.setVerbosity(0)

  # useful constants
  rLen = len(R)
  M = 10000 # a large number
  Sr = range(len(S))
  Ar = range(len(A))
  
  # decision variables
  # FIXME i removed upper bound of x. it shoundn't have such bound without transient-state assumption, right?
  x = m.new((len(S), len(A)), lb=0, name='x')
  z = m.new(rLen, vtype=bool, name='z')
  y = m.new(rLen, name='y')

  # constraints on y
  m.constrain([y[i] <= sum([x[s, a] * R[i](S[s], A[a]) for s in Sr for a in Ar]) - maxV[i] + (1 - z[i]) * M for i in xrange(rLen)])
  m.constrain([y[i] <= z[i] * M for i in xrange(rLen)])
  
  # constraints on x (valid occupancy)
  for sp in Sr:
    if S[sp] == s0:
      m.constrain(sum([x[sp, ap] for ap in Ar]) == 1)
    else:
      m.constrain(sum([x[sp, ap] for ap in Ar]) == sum([x[s, a] * T(S[s], A[a], S[sp]) for s in Sr for a in Ar]))
  
  # obj
  obj = m.maximize(sum([psi[i] * y[i] for i in xrange(rLen)]))

  if config.VERBOSE:
    print 'obj', obj
    print 'x', m[x]
    print 'y', m[y]
    print 'z', m[z]
  
  # build occupancy as S x A -> x[.,.]
  # z[i] == 1 then this policy is better than maxV on the i-th reward candidate
  res = util.Counter()
  for s in Sr:
    for a in Ar:
      res[S[s], A[a]] = m[x][s, a] 
  return res

def domPiMilp(S, A, r, T, s0, terminal, domPis, consIdx, gamma=1):
  """
  Finding dominating policies by representing constraints as possible negative rewards.
  Described in the report on aug.19, 2017.
  """
  rmax = 10000
  M = 0.001
  consLen = len(consIdx)

  m = CPlexModel()
  if not config.VERBOSE: m.setVerbosity(0)

  # state range
  Sr = range(len(S))
  # action range
  Ar = range(len(A))
  
  # decision variables
  x = m.new((len(S), len(A)), lb=0, name='x')
  z = m.new(consLen, vtype=bool, name='z')
  #z = [0, 1, 0] # test for office nav domain
  t = m.new(name='t')
  
  # flow conservation
  for sp in Sr:
    # x (x(s) - \gamma * T) = \sigma
    # and make sure there is no flow back from the terminal states
    if not terminal(S[sp]):
      m.constrain(sum(x[s, a] * ((s == sp) - gamma * T(S[s], A[a], S[sp]) * (not terminal(S[s]))) for s in Sr for a in Ar) == (S[sp] == s0))
      #print S[sp], [(S[s], A[a]) for s in Sr for a in Ar if T(S[s], A[a], S[sp]) > 0]
    else:
      m.constrain(sum(x[s, a] * ((s == sp) - gamma * T(S[s], A[a], S[sp]) * (not terminal(S[sp]))) for s in Sr for a in Ar) == (S[sp] == s0))

  # t is the lower bound of the difference between x and y
  # note: i don't think expressions in constraints can call other functions
  for y in domPis:
    # note that y is indexed by elements in S x A, not numbered indices
    m.constrain(sum(x[s, a] * r(S[s], A[a]) for s in Sr for a in Ar) -\
                sum(y[S[s], A[a]] *
                    (r(S[s], A[a]) + sum(- rmax * (S[s][consIdx[i]] != s0[consIdx[i]]) * z[i] for i in range(consLen)))\
                    for s in Sr for a in Ar)\
                >= t)
   
  for s in Sr:
    for i in range(consLen):
      if S[s][consIdx[i]] != s0[consIdx[i]]:
        for a in Ar:
          m.constrain(z[i] + M * x[s, a] <= 1)

  # obj
  obj = m.maximize(t)
  
  print m[z]
  
  return obj, {(S[s], A[a]): m[x][s, a] for s in Sr for a in Ar}
  
def rewardUncertainMILP(S, A, R, T, s0, terminal, k, optV, gamma=1):
  """
  The algorithm adapted from
  Viappiani, Paolo and Boutilier, CraigOptimal. set recommendations based on regret

  This algorithm would find the minimax-regret policy query in our problem.
  Not sure how to use this algorithm.
  """
  m = CPlexModel()
  if not config.VERBOSE: m.setVerbosity(0)

  M = 100000

  # state range
  Sr = range(len(S))
  # action range
  Ar = range(len(A))
  
  mr = m.new(name='mr')
  # decision variables
  x = m.new((k, len(S), len(A)), lb=0, name='x')
  v = m.new((k, len(R)), name='v')
  I = m.new((k, len(R)), vtype=bool, name='I')
  
  for r in range(len(R)):
    m.constrain(mr >= sum(v[i, r]) for i in range(k))
  
  for r in range(len(R)):
    for i in range(k):
      m.constrain(v[i, r] >= optV[r] - sum(x[i, s, a] * R[r](S[s], A[a]) for s in Sr for a in Ar) + (I[i, r] - 1) * M)

  # make sure x is a valid occupancy
  for i in range(k):
    for sp in Sr:
      m.constrain(sum(x[i, s, a] * ((s == sp) - gamma * T(S[s], A[a], S[sp])) for s in Sr for a in Ar) == (S[sp] == s0))
  
  for r in range(len(R)):
    m.constrain(sum(I[i, r] for i in range(k)) == 1)
  
  for r in range(len(R)):
    for i in range(k):
      m.constrain(v[i, r] >= 0)
  
  obj = m.minimize(mr)
  
  return obj, m[I]

def computeObj(q, psi, S, A, R):
  rLen = len(R)
  obj = 0

  for i in xrange(rLen):
    values = [computeValue(pi, R[i], S, A) for pi in q]
    obj += psi[i] * max(values)
    #print filter(lambda _: values[_] == max(values), range(len(q))), max(values)
  
  return obj

def computeValue(pi, r, S, A):
  sum = 0

  if pi == {}:
    return sum
  else:
    for s in S:
      for a in A:
        sum += pi[s, a] * r(s, a)
    return sum

def toyDomain():
  args = easyDomains.getChainDomain(10)
  args['maxV'] = [0]
  milp(**args)


if __name__ == '__main__':
  config.VERBOSE = True

  #rockDomain()
  toyDomain()
