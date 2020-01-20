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

def lpDualGurobi(mdp, zeroConstraints=(), positiveConstraints=(), positiveConstraintsOcc=0, unknownStateCons=(), violationCost=None):
  """
  Solve the dual problem of lp.
  This function is overridden. If violationCost is not None, then we punish the robot by changing an unknown feature.
  Otherwise unknown features are imposed as hard constraints.

  :param violationCost: if not None, it's the cost of violating a constraint rather than enforcing it.
  :return: {'feasible': if a feasible solution is found,
            'obj': the objective value,
            'pi': the (safely-)optimal policy}
  """
  if len(positiveConstraints) == 0 and positiveConstraintsOcc > 0:
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

  # use integer variables to indicate which constraints are violated
  # not going to returned this though.. the indices are wrong (we excluded known-to-be-locked/free features)
  zC = m.addVars(len(unknownStateCons), vtype=GRB.BINARY, name='zC')

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

  # >= constraints. the occupancy should be at least positiveConstraintsOcc
  if len(positiveConstraints) > 0:
    #FIXME positiveConstraints still have actions in them. in consistent with other types of constraints
    m.addConstr(sum(x[S.index(s), A.index(a)] for s, a in positiveConstraints) >= positiveConstraintsOcc)
    
  if len(zeroConstraints) > 0:
    for consIdx in range(len(zeroConstraints)):
      m.addConstr(sum(x[S.index(s), A.index(a)] for s in zeroConstraints[consIdx] for a in A) == 0)

  # add cost of queries
  if len(unknownStateCons) > 0:
    for consIdx in range(len(unknownStateCons)):
      m.addConstr(M * zC[consIdx] >= sum(x[S.index(s), A.index(a)] for s in unknownStateCons[consIdx] for a in A))

  # obj
  m.setObjective(sum([x[s, a] * r(S[s], A[a]) for s in Sr for a in Ar])
                 - sum(zC[consIdx] * violationCost for consIdx in range(len(unknownStateCons))),
                 GRB.MAXIMIZE)

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

def lpDualCPLEX(mdp, zeroConstraints=(), positiveConstraints=(), positiveConstraintsOcc=1):
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

def milp(mdp, maxV, zeroConstraints=()):
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

  # == constraints
  for consIdx in range(len(zeroConstraints)):
    m.addConstr(sum(x[S.index(s), A.index(a)] for s, a in zeroConstraints[consIdx]) == 0)
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

def jointUncertaintyMilp(mdp, oldPi, oldZC, unknownFeatStates, costOfQuery):
  """
  Find the MILP formulation in report 12/1/2019.
  It finds the second policy in a batch policy query, considering when it outperforms the previous policy in terms of
  reward functions and changeability of unknown features.

  :param mdp: the transition function of this mdp should have be revised to encode p_f
  :param oldPi: the first policy that is added.
  :param lockedFeatStates: the states which the robot should not visit. They change some known-to-be-locked features.
  :param unknownFeatStates: the states which the robot may change, but need to pay the query cost.
  :param costOfQuery: the cost of querying.
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
  y = m.addVars(rLen, name='y')
  # y prime, a helper variable
  y0 = m.addVars(rLen, name='y0', lb=0)

  # oldPi is a mapping from state, action (in S x A) to occupancy
  # to be consistent with x, convert it to a mapping from (s, a) where s in Sr, a in Ar
  oldX = {(s, a): oldPi[S[s], A[a]] for s in Sr for a in Ar}

  # integer variables
  zR = m.addVars(rLen, vtype=GRB.BINARY, name='zR')
  zC = m.addVars(len(unknownFeatStates), vtype=GRB.BINARY, name='zC')
  # zCNew indicates the newly changed features by x. note that it does not need to be constrained as integers
  zCNew = m.addVars(len(unknownFeatStates), lb=0, name='zCNew')

  zSafe = m.addVar(vtype=GRB.BINARY, name='zSafe')

  V = lambda x_local, r: sum([x_local[s, a] * r(S[s], A[a]) for s in Sr for a in Ar])

  # (a) flow conservation constraint
  for sp in Sr:
    m.addConstr(sum(x[s, a] * ((s == sp) - gamma * T(S[s], A[a], S[sp])) for s in Sr for a in Ar) == alpha(S[sp]))

  # (b) is encoded in the transition function

  for consIdx in range(len(unknownFeatStates)):
    # (c) unknown features can be changed
    m.addConstr(M * zC[consIdx] >= sum(x[S.index(s), A.index(a)] for s in unknownFeatStates[consIdx] for a in A))
    # (d) constrain z^{new}_\phi
    m.addConstr(zCNew[consIdx] >= zC[consIdx] - oldZC[consIdx])

  # (e) constraints on y^0_r
  m.addConstr(sum(zC[idx] for idx in range(len(oldZC)) if oldZC[idx] == 1) <= sum(oldZC) - 1 + zSafe * M)
  for i in range(rLen):
    m.addConstr(y0[i] >= V(oldX, R[i]) - (1 - zSafe) * M)

  # (f) constraints on y_r
  for i in range(rLen):
    m.addConstr(y[i] <= V(x, R[i]) - y0[i] + (1 - zR[i]) * M)
    m.addConstr(y[i] <= 0 + zR[i] * M)

  # obj
  m.setObjective(sum([psi[i] * y[i] for i in xrange(rLen)])
                 - sum(zC[idx] * costOfQuery for idx in range(len(unknownFeatStates))),
                 GRB.MAXIMIZE)

  m.optimize()

  pi = {(S[s], A[a]): x[s, a].X for s in Sr for a in Ar}

  if config.VERBOSE:
    # print decision variables other than pi for debugging
    print 'oldZC', oldZC
    print 'zC', [zC[consIdx].X for consIdx in range(len(unknownFeatStates))]
    print 'y0 values', [y0[rIdx].X for rIdx in range(rLen)]
    print 'y values', [y[rIdx].X for rIdx in range(rLen)]

  if m.status == GRB.Status.OPTIMAL:
    # return feasible being true and the obj value, opt pi
    # .X attribute is to retrieve the value of the variable
    return pi
  else:
    # simply return infeasible
    raise Exception('milp problem optimal solution not found' + m.status)

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
