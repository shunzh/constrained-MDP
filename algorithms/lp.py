import config
import util

if config.OPT_METHOD == 'gurobi':
  from gurobipy import *
elif config.OPT_METHOD == 'cplex':
  from pycpx import CPlexModel, CPlexException, CPlexNoSolution
else:
  raise Exception('unknown optimization method ' + config.OPT_METHOD)

def linearRegression(A, b):
  """
  find min_x ||Ax - b||^2
  """
  m = Model()
  m.setParam('OutputFlag', False)

  n = len(A) # number of rows in A
  d = len(A[0]) # number of columns in A
  assert n == len(b) # make sure the shape of matrix is correct

  # x is of size d
  x = m.addVars(d, name='x')
  # ** is not supported by gurobi!
  square = lambda _: _ * _
  # \sum_i (A[i] * x - b[i])^2
  m.setObjective(sum(square(sum(A[i][j] * x[j] for j in xrange(d)) - b[i])
                     for i in xrange(n)), GRB.MINIMIZE)
  m.optimize()

  return [x[_].X for _ in xrange(d)]

def lpDualGurobi(mdp, zeroConstraints=(), positiveConstraints=(), positiveConstraintsOcc=0, violationCost=None):
  """
  Solve the dual problem of lp, maybe with some constraints
  :param violationCost: if not None, it's the cost of violating a constraints rather than enforcing it.

  Note that this is a lower level function that does not consider feature extraction.
  r should be a reward function, not a reward parameter.
  """
  if len(positiveConstraints) == 0 and positiveConstraintsOcc > 0:
    print 'returned here'
    return {'feasible': False}

  S = mdp.S
  A = mdp.A
  T = mdp.T
  r = mdp.r
  gamma = mdp.gamma
  alpha = mdp.alpha
  terminal = mdp.terminal
  
  # initialize a Gurobi model
  m = Model()
  m.setParam('OutputFlag', False)

  # useful constants
  Sr = range(len(S))
  Ar = range(len(A))

  x = m.addVars(len(S), len(A), lb=0, name='x')

  M = 10000  # a large number
  zC = m.addVars(len(zeroConstraints), vtype=GRB.BINARY, name='zC')

  nonTerminalStatesRange = filter(lambda _: not terminal(S[_]), Sr)

  # flow conservation constraints. for each s',
  # \sum_{s, a} x(s, a) (1_{s = s'} - \gamma * T(s, a, s')) = \alpha(s')
  if mdp.invertT is not None:
    # if invertT is computed for deterministic domains, this can be much more efficient
    # sp ('next state' in the transition) are non-terminal states
    for sp in nonTerminalStatesRange:
      # supports of 1_{s = s'}
      identityItems = [(sp, a) for a in Ar]
      # supports of \gamma * T(s, a, s')
      invertedTransitItems = map(lambda _: (S.index(_[0]), A.index(_[1])), mdp.invertT[S[sp]])
      # kill duplicates
      supports = tuple(set(identityItems).union(invertedTransitItems))
      m.addConstr(sum(x[s, a] * ((s == sp) - gamma * T(S[s], A[a], S[sp])) for (s, a) in supports) == alpha(S[sp]))
  else:
    # the normal way, exactly as specified in the formula
    # note that we need to iterate overall state, action pairs for each s' \in S
    for sp in nonTerminalStatesRange:
      m.addConstr(sum(x[s, a] * ((s == sp) - gamma * T(S[s], A[a], S[sp])) for s in nonTerminalStatesRange for a in Ar) == alpha(S[sp]))

  # == constraints
  if len(zeroConstraints) > 0:
    if violationCost is None:
      m.addConstr(sum(x[S.index(s), A.index(a)] for s, a in zeroConstraints) == 0)
    else:
      m.addConstr(M * zC >= sum(x[S.index(s), A.index(a)] for s, a in zeroConstraints))

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
    return {'feasible': False, 'obj': 0, 'pi': None}
  else:
    raise Exception('error status: %d' % m.status)

def lpDualCPLEX(mdp, zeroConstraints=[], positiveConstraints=[], positiveConstraintsOcc=1):
  """
  DEPRECATED since we moved to gurobi. but leave the function here for sanity check

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


def milp(mdp, maxV):
  """
  Solve the MILP problem in greedy construction of policy query

  :param maxV maxV[i] = max_{\pi \in q} V_{r_i}^\pi
  """
  m = Model()
  m.setParam('OutputFlag', False)

  # convert notation to previous implementation
  S = mdp.S
  A = mdp.A
  R = mdp.rFuncs
  psi = mdp.psi
  T = mdp.T
  alpha = mdp.alpha
  gamma = mdp.gamma

  # useful constants
  rLen = len(R)
  M = 10000  # a large number
  Sr = range(len(S))
  Ar = range(len(A))

  # decision variables
  x = m.addVars(len(S), len(A), lb=0, name='x')
  z = m.addVars(rLen, vtype=GRB.BINARY, name='z')
  y = m.addVars(rLen, name='y')

  # constraints on y
  for i in range(rLen):
    m.addConstr(y[i] <= sum([x[s, a] * R[i](S[s], A[a]) for s in Sr for a in Ar]) - maxV[i] + (1 - z[i]) * M)
    m.addConstr(y[i] <= z[i] * M)

  # constraints on x (valid occupancy)
  for sp in Sr:
    m.addConstr(sum(x[s, a] * ((s == sp) - gamma * T(S[s], A[a], S[sp])) for s in Sr for a in Ar) == alpha(S[sp]))

  # obj
  m.setObjective(sum([psi[i] * y[i] for i in xrange(rLen)]), GRB.MAXIMIZE)

  m.optimize()

  pi = {(S[s], A[a]): x[s, a].X for s in Sr for a in Ar}

  if m.status == GRB.Status.OPTIMAL:
    # return feasible being true and the obj value, opt pi
    # .X attribute is to retrieve the value of the variable
    return pi
  else:
    # simply return infeasible
    raise Exception('milp problem optimal solution not found' + m.status)


#TODO the following functions still use CPLEX.
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

def decomposePiLP(S, A, T, s0, terminal, rawX, x, gamma=1):
  """
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

"""
Utility functions to compute values, uncertain objectives, etc.
"""
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
