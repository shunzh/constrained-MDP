import copy
import getopt
import pickle
import random
import sys
import time

import numpy
import scipy

import config
from algorithms.consQueryAgents import ConsQueryAgent, EXIST, NOTEXIST
from algorithms.initialSafeAgent import OptQueryForSafetyAgent, GreedyForSafetyAgent, \
  MaxProbSafePolicyExistAgent, DomPiHeuForSafetyAgent, DescendProbQueryForSafetyAgent, OracleSafetyAgent
from algorithms.jointUncertaintyAgents import JointUncertaintyQueryByMyopicSelectionAgent, \
  JointUncertaintyOptimalQueryAgent, JointUncertaintyRandomQuery, \
  JointUncertaintyQueryBySamplingDomPisAgent
from algorithms.jointUncertaintyBatchQueryAgent import JointUncertaintyBatchQueryAgent
from algorithms.safeImprovementAgent import SafeImproveAgent
from domains.officeNavigation import officeNavigationTask, squareWorld, carpetsAndWallsDomain
from util import normalize, printOccSA


def saveData(results, rnd, prefix=''):
  """
  Save results to [rnd].pkl
  If prior results exist, update it
  """
  filename = str(rnd) + '.pkl'
  pickle.dump(results, open(prefix + filename, 'wb'))

def findInitialSafePolicy(mdp, consStates, goalStates, trueFreeFeatures, rnd, consProbs=None):
  queries = {}
  valuesOfSafePis = {}
  times = {}
  # these are assigned when ouralg is run
  iiss = None
  relFeats = None

  # keep track of algorithms' answers on whether problems are solvable
  answer = None
  thisAnswer = None

  from config import methods

  for method in methods:
    print method
    queries[method] = []
    times[method] = []

    # ======== timed session starts ========
    start = time.time()

    # oracle / opt
    if method == 'oracle':
      agent = OracleSafetyAgent(mdp, consStates, trueFreeFeatures, goalStates=goalStates, consProbs=consProbs)
    elif method == 'opt':
      agent = OptQueryForSafetyAgent(mdp, consStates, goalStates=goalStates, consProbs=consProbs)
    elif method == 'optLocked':
      agent = OptQueryForSafetyAgent(mdp, consStates, goalStates=goalStates, consProbs=consProbs, optimizeLocked=True,
                                     optimizeFree=False)
    elif method == 'optFree':
      agent = OptQueryForSafetyAgent(mdp, consStates, goalStates=goalStates, consProbs=consProbs, optimizeLocked=False,
                                     optimizeFree=True)

    # our heuristics
    elif method == 'iisAndRelpi':
      agent = GreedyForSafetyAgent(mdp, consStates, goalStates=goalStates, consProbs=consProbs)
    elif method == 'iisAndRelpi1':
      # only with extended belief
      agent = GreedyForSafetyAgent(mdp, consStates, goalStates=goalStates, consProbs=consProbs, heuristicID=1)
    elif method == 'setcoverNonBayes':
      agent = GreedyForSafetyAgent(mdp, consStates, goalStates=goalStates, consProbs=None, useIIS=True, useRelPi=True)
    elif method == 'setcoverWithValue':
      agent = GreedyForSafetyAgent(mdp, consStates, goalStates=goalStates, consProbs=consProbs, optimizeValue=True)
    elif method == 'iisOnly':
      agent = GreedyForSafetyAgent(mdp, consStates, goalStates=goalStates, consProbs=consProbs, useIIS=True,
                                   useRelPi=False)
    elif method == 'relpiOnly':
      agent = GreedyForSafetyAgent(mdp, consStates, goalStates=goalStates, consProbs=consProbs, useIIS=False,
                                   useRelPi=True)

    elif method == 'iisAndRelpi2':
      # with extended belief and submodular estimate
      agent = GreedyForSafetyAgent(mdp, consStates, goalStates=goalStates, consProbs=consProbs, heuristicID=2)

    elif method == 'iisAndRelpi3':
      agent = GreedyForSafetyAgent(mdp, consStates, goalStates=goalStates, consProbs=consProbs, heuristicID=3)
    elif method == 'iisOnly3':
      agent = GreedyForSafetyAgent(mdp, consStates, goalStates=goalStates, consProbs=consProbs, useIIS=True,
                                   useRelPi=False, heuristicID=3)
    elif method == 'relpiOnly3':
      agent = GreedyForSafetyAgent(mdp, consStates, goalStates=goalStates, consProbs=consProbs, useIIS=False,
                                   useRelPi=True, heuristicID=3)

    elif method == 'iisAndRelpi4':
      agent = GreedyForSafetyAgent(mdp, consStates, goalStates=goalStates, consProbs=consProbs, heuristicID=4)

    # baseline heuristics
    elif method == 'maxProb':
      agent = MaxProbSafePolicyExistAgent(mdp, consStates, goalStates=goalStates, consProbs=consProbs, tryFeasible=True,
                                          tryInfeasible=True)
    elif method == 'maxProbF':
      agent = MaxProbSafePolicyExistAgent(mdp, consStates, goalStates=goalStates, consProbs=consProbs, tryFeasible=True,
                                          tryInfeasible=False)
    elif method == 'maxProbIF':
      agent = MaxProbSafePolicyExistAgent(mdp, consStates, goalStates=goalStates, consProbs=consProbs,
                                          tryFeasible=False, tryInfeasible=True)
    elif method == 'piHeu':
      agent = DomPiHeuForSafetyAgent(mdp, consStates, goalStates=goalStates, consProbs=consProbs)
    elif method == 'piHeuWithValue':
      agent = DomPiHeuForSafetyAgent(mdp, consStates, goalStates=goalStates, consProbs=consProbs, optimizeValue=True)
    elif method == 'random':
      agent = DescendProbQueryForSafetyAgent(mdp, consStates, goalStates=goalStates, consProbs=consProbs)
    else:
      raise Exception('unknown method', method)

    if iiss is None and hasattr(agent, 'iiss'):
      iiss = agent.iiss
    if relFeats is None and hasattr(agent, 'relFeats'):
      relFeats = agent.domPiFeats

    # it should not query more than the number of total features anyway..
    # but in case of bugs, this should not be a dead loop
    while len(queries[method]) < len(consStates) + 1:
      query = agent.findQuery()

      if query == EXIST or query == NOTEXIST:
        # the agent stops querying
        thisAnswer = query
        break
      elif query in trueFreeFeatures:
        agent.updateFeats(newFreeCon=query)
      else:
        agent.updateFeats(newLockedCon=query)

      queries[method].append(query)

    end = time.time()
    # ======== timed session ends ========

    # the alg must return an answer
    assert thisAnswer is not None

    # make sure all the algorithms give the same answer. otherwise imp error
    if answer is None:
      answer = thisAnswer
    else:
      assert answer == thisAnswer, {'other methods say': answer, method + ' says': thisAnswer}

    # make sure that, if safe policy exists, safe policy found
    if thisAnswer == EXIST:
      # may use other ways? most algorithms check this before returning anyway
      assert agent.safePolicyExist()
      valuesOfSafePis[method] = agent.safePolicyValue()

    # FIXME why list?
    times[method].append(end - start)

  print 'queries', queries
  print 'times', times
  print 'safe policy', answer
  # print 'safe policy value', valuesOfSafePis

  if 'oracle' in queries.keys() and len(queries['oracle']) == 0:
    # do not record cases where it is impossible to find a safe policy (because of the transition dynamics)
    return

def improveSafePolicyMMR(mdp, consStates, k, rnd):
  agent = SafeImproveAgent(mdp, consStates)

  # find dom pi (which may be used to find queries and will be used for evaluation)
  start = time.time()
  relFeats, domPis = agent.findRelevantFeaturesAndDomPis()
  end = time.time()
  domPiTime = end - start

  methods = ['alg1', 'chain', 'naiveChain', 'relevantRandom', 'random', 'nq']

  # decide the true changeable features for expected regrets
  numpy.random.seed(2 * (1 + rnd))  # avoid weird coupling, e.g., the ones that are queried are exactly the true changeable ones
  if len(agent.unknownCons) < k:
    raise Exception('k is larger than the number of unknown features so no need to select queries. abort.')
  violableIndices = numpy.random.choice(range(len(agent.unknownCons)), k, replace=False)
  violableCons = [agent.unknownCons[_] for _ in violableIndices]

  for method in methods:
    start = time.time()
    if method == 'brute':
      q = agent.findMinimaxRegretConstraintQBruteForce(k, relFeats, domPis)
    elif method == 'reallyBrute':
      # really brute still need domPis to find out MR...
      q = agent.findMinimaxRegretConstraintQBruteForce(k, agent.unknownCons, domPis)
    elif method == 'alg1':
      q = agent.findMinimaxRegretConstraintQ(k, relFeats, domPis)
    elif method == 'alg1NoFilter':
      q = agent.findMinimaxRegretConstraintQ(k, relFeats, domPis, filterHeu=False)
    elif method == 'alg1NoScope':
      q = agent.findMinimaxRegretConstraintQ(k, relFeats, domPis, scopeHeu=False)
    elif method == 'naiveChain':
      q = agent.findChainedAdvConstraintQ(k, relFeats, domPis, informed=False)
    elif method == 'chain':
      q = agent.findChainedAdvConstraintQ(k, relFeats, domPis, informed=True)
    elif method == 'relevantRandom':
      q = agent.findRelevantRandomConstraintQ(k, relFeats)
    elif method == 'random':
      q = agent.findRandomConstraintQ(k)
    elif method == 'nq':
      q = []
    elif method == 'domPiBruteForce':
      # HACKING compute how long is needed to find a dominating policies by enumeration
      agent.findRelevantFeaturesBruteForce()
      q = []
    else:
      raise Exception('unknown method', method)
    end = time.time()

    # note that we compute domPiTime in the begining to avoid recompute it for every alg
    # some alg actually does not need dom pis
    runTime = end - start + (0 if method in ['random', 'nq'] else domPiTime)

    print method, q

    mrk, advPi = agent.findMRAdvPi(q, relFeats, domPis, k, consHuman=True)

    regret = agent.findRegret(q, violableCons)

    print mrk, regret, runTime

def jointUncertaintyQuery(mdp, method, consStates, consProbs, trueRewardIdx, trueFreeFeatures, rnd, costOfQuery):
  """
  Query under both reward uncertainty and safety constraint uncertainty.

  For now, assume initial safe policies exist and the robot can pose at most k queries
  """
  results = {}

  # the agent is going to modify mdp.psi, so make copies here
  mdpForAgent = copy.deepcopy(mdp)
  queriesAsked = []

  start = time.time()

  if method == 'opt':
    agent = JointUncertaintyOptimalQueryAgent(mdpForAgent, consStates, consProbs=consProbs, costOfQuery=costOfQuery)
  elif method == 'myopic':
    agent = JointUncertaintyQueryByMyopicSelectionAgent(mdpForAgent, consStates, consProbs=consProbs, costOfQuery=costOfQuery)
  elif method == 'myopicReward':
    agent = JointUncertaintyQueryByMyopicSelectionAgent(mdpForAgent, consStates, consProbs=consProbs,
                                                        costOfQuery=costOfQuery, heuristic='rewardFirst')
  elif method == 'myopicFeature':
    agent = JointUncertaintyQueryByMyopicSelectionAgent(mdpForAgent, consStates, consProbs=consProbs,
                                                        costOfQuery=costOfQuery, heuristic='featureFirst')
  elif method == 'batch':
    agent = JointUncertaintyBatchQueryAgent(mdpForAgent, consStates, consProbs=consProbs, costOfQuery=costOfQuery, qi=True)
  elif method == 'dompi':
    agent = JointUncertaintyQueryBySamplingDomPisAgent(mdpForAgent, consStates, consProbs=consProbs,
                                                       costOfQuery=costOfQuery, heuristicID=0)
  elif method == 'dompiUniform':
    agent = JointUncertaintyQueryBySamplingDomPisAgent(mdpForAgent, consStates, consProbs=consProbs,
                                                       costOfQuery=costOfQuery, heuristicID=1)
  elif method == 'random':
    agent = JointUncertaintyRandomQuery(mdpForAgent, consStates, consProbs=consProbs, costOfQuery=costOfQuery)
  else:
    raise Exception('unknown method ' + str(method))

  while True:
    query = agent.findQuery()

    if query is None:
      break
    else:
      queriesAsked.append(query)
      (qType, qContent) = query
      if config.VERBOSE: print 'QUERY POSED', query

      if qType == 'F':
        # a feature query
        if qContent in trueFreeFeatures:
          agent.updateFeats(newFreeCon=qContent)
        else:
          agent.updateFeats(newLockedCon=qContent)
      elif qType == 'R':
        if trueRewardIdx in qContent:
          agent.updateReward(consistentRewards=qContent)
        else:
          agent.updateReward(inconsistentRewards=qContent)
      else:
        raise Exception('unknown qType ' + qType)

  end = time.time()
  duration = end - start

  # find the robot's policy, and evaluate under the true reward function
  # this is for fair comparision between agents, since different agents have different reward beliefs
  agentPi = agent.computeCurrentSafelyOptPi()
  # set the reward function in-place, we are not going to use agent after this anyway
  agent.updateReward(consistentRewards=[trueRewardIdx])
  value = agent.computeValue(agentPi)

  print 'RESULTS: rnd', rnd, method,\
    'obj', value - len(queriesAsked) * costOfQuery,\
    'value', value,\
    'queries', queriesAsked,\
    'time', duration
  print

  results = {'value': value, 'queries': queriesAsked, 'time': duration}
  return results

def experiment(mdp, consStates, goalStates, k, rnd, pf=0, pfStep=1, costOfQuery=0.0):
  """
  Find queries to find initial safe policy or to improve an existing safe policy.

  k: number of queries (in batch querying setting
  dry: no output to file if True
  rnd: random seed
  pf: only for Bayesian setting. ["prob that ith unknown feature is free" for i in range(self.numOfCons)]
    If None (by default), set randomly
  """
  numOfCons = len(consStates)
  numOfRewards = len(mdp.psi)

  # consProbs is None then it's Bayesian setting, otherwise MMR
  consProbs = [pf + pfStep * random.random() for _ in range(numOfCons)]
  print 'consProbs', zip(range(numOfCons), consProbs)

  from config import sampleInstances, methods

  keys = ['value', 'queries', 'time']

  batch_results = {}
  for method in methods:
    batch_results[method] = {key: [] for key in keys}

  for instanceIdx in range(sampleInstances):
    # true free features, randomly generated
    trueFreeFeatures = filter(lambda idx: random.random() < consProbs[idx], range(numOfCons))
    trueRewardFuncIdx = numpy.random.choice(range(numOfRewards), p=mdp.psi)

    for method in methods:
      print 'true free features', trueFreeFeatures
      print 'true reward function index', trueRewardFuncIdx

      # under joint uncertainty:
      results = jointUncertaintyQuery(mdp, method, consStates, consProbs, trueRewardFuncIdx, trueFreeFeatures, rnd, costOfQuery)

      for key in keys:
        batch_results[method][key].append(results[key])

  return batch_results

def setRandomSeed(rnd):
  print 'random seed', rnd
  random.seed(rnd)
  numpy.random.seed(rnd)
  scipy.random.seed(rnd)

if __name__ == '__main__':
  # default values
  method = None
  k = 5 # DUMMY for joint uncertainty experiments

  # the domain is size x size
  size = 6

  # these should be set in config
  numOfCarpets = None
  numOfSwitches = None

  numOfWalls = 5
  from config import trialsStart, trialsEnd, numsOfCarpets, numsOfSwitches, costsOfQuery

  rnd = 0 # set a dummy random seed if no -r argument
  dry = False # no output to files if dry run

  try:
    opts, args = getopt.getopt(sys.argv[1:], 'm:k:n:s:r:R:dp:v')
  except getopt.GetoptError:
    raise Exception('Unknown flag')
  for opt, arg in opts:
    if opt == '-k':
      k = int(arg)
    elif opt == '-m':
      size = int(arg)

    # only run specific set-ups
    elif opt == '-n':
      numsOfCarpets = [int(arg)]
    elif opt == '-s':
      numsOfSwitches = [int(arg)]

    elif opt == '-r':
      rnd = int(arg)

      trialsStart = rnd
      trialsEnd = rnd + 1
    elif opt == '-R':
      # running starting from trialsStart
      trialsStart = int(arg)
    elif opt == '-d':
      dry = True
    elif opt == '-v':
      config.VERBOSE = True
    else:
      raise Exception('unknown argument')

  for rnd in range(trialsStart, trialsEnd):
    results = {}
    for numOfCarpets in numsOfCarpets:
      for numOfSwitches in numsOfSwitches:
        for costOfQuery in costsOfQuery:
          setRandomSeed(rnd)
          print '# of carpets:', numOfCarpets, '# of switches:', numOfSwitches, 'query cost', costOfQuery

          #spec = carpetsAndWallsDomain(); numOfSwitches = len(spec.switches)
          spec = squareWorld(size=size, numOfCarpets=numOfCarpets, numOfWalls=numOfWalls, numOfSwitches=numOfSwitches)

          # uniform prior over rewards
          #rewardProbs = [1.0 / numOfSwitches] * numOfSwitches
          # random prior over rewards (add 0.1 to reduce variance a little bit)
          rewardProbs = normalize([random.random() for _ in range(numOfSwitches)]); print 'psi', rewardProbs

          mdp, consStates, goalStates = officeNavigationTask(spec, rewardProbs=rewardProbs, gamma=0.99)
          results[(numOfCarpets, numOfSwitches, costOfQuery)] = experiment(mdp, consStates, goalStates, k, rnd, costOfQuery=costOfQuery)

    if not dry: saveData(results, rnd)
