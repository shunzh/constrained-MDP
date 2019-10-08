from QTPAgent import QTPAgent
import util
import config
import numpy
import lp
import copy


class GreedyConstructionPiAgent(QTPAgent):
  def __init__(self, mdp, queryType):
    """
    qi: query iteration
    """
    if hasattr(self, 'computePiValue'):
      # policy gradient agent has different ways to compute values..
      self.computeV = lambda pi, S, A, r, horizon: self.computePiValue(pi, r, horizon)
    else:
      self.computeV = lambda pi, S, A, r, horizon: lp.computeValue(pi, r, S, A)

  def learn(self):
    self.args = args  # save a copy
    horizon = self.cmp.horizon
    terminalReward = self.cmp.terminalReward

    if self.queryType == QueryType.ACTION:
      k = len(args['A'])
    else:
      k = config.NUMBER_OF_RESPONSES

    # now q is a set of policy queries
    bestQ = None
    bestEUS = -numpy.inf

    # keep a copy of currently added policies. may not be used.
    # note that this is passing by inference

    # start with the prior optimal policy
    q = [self.getFiniteVIAgent(self.phi, horizon, terminalReward, posterior=True).x]
    args['q'] = q
    objValue = None  # k won't be 1, fine

    # start adding following policies
    for i in range(1, k):
      if config.VERBOSE: print 'iter.', i
      x = self.findNextPolicy(**args)
      q.append(x)

      args['q'] = q

    if self.queryType == QueryType.POLICY:
      # if asking policies directly, then return q
      # return q, objValue # THIS RETURNS EUS, NOT EPU
      return q, objValue
    if self.queryType == QueryType.PARTIAL_POLICY:
      idx = 0
      objValue = self.getQValue(self.cmp.state, None, q)
      qP = copy.copy(q)

      while True:
        # iterate over all the policies, remove one state pair of each
        # but make sure the EUS of the new set is unchaged
        x = qP[idx]
        xOld = x.copy()

        success = False
        for key in util.randomly(x.keys()):
          x.pop(key)
          print self.getQValue(self.cmp.state, None, qP), objValue
          if self.getQValue(self.cmp.state, None, qP) == objValue:
            success = True
            break
          else:
            x = xOld.copy()

        if not success: break
        # print idx, len(x)
        idx = (idx + 1) % len(q)

      return qP
    elif self.queryType == QueryType.DEMONSTRATION:
      # if we already build a set of policies, but the query type is demonstration
      # we sample trajectories from these policies as a query
      # note that another way is implemented in MILPDemoAgent, which choose the next policy based on the demonstrated trajectories.
      qu = [self.sampleTrajectory(x) for x in q]
      return qu
    elif self.queryType in [QueryType.SIMILAR, QueryType.ACTION]:
      # implemented in a subclass, do nothing here
      pass
    else:
      raise Exception('Query type not implemented for MILP.')

    return args, q


class MILPAgent(GreedyConstructionPiAgent):
  def findNextPolicy(self, S, A, R, T, s0, psi, q):
    rewardCandNum = len(self.rewardSet)
    horizon = self.cmp.horizon

    maxV = []
    for rewardId in xrange(rewardCandNum):
      maxV.append(max([self.computeV(pi, S, A, R[rewardId], horizon) for pi in q]))

    # solve a MILP problem
    return lp.milp(S, A, R, T, s0, psi, maxV)

