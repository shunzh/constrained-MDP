import getopt
import os
import pickle
import random
import sys
import time

import numpy
import scipy

from algorithms.consQueryAgents import ConsQueryAgent, EXIST, NOTEXIST
from algorithms.initialSafeAgent import OptQueryForSafetyAgent, GreedyForSafetyAgent, \
  MaxProbSafePolicyExistAgent, DomPiHeuForSafetyAgent, DescendProbQueryForSafetyAgent
from algorithms.safeImprovementAgent import SafeImproveAgent
from domains.officeNavigation import officeNavigation, squareWorld, toySokobanWorld, sokobanWorld, carpetsAndWallsDomain


def experiment(spec, k, constrainHuman, dry, rnd, gamma=0.9, pf=0, pfStep=1):
  """
  Find queries to find initial safe policy or to improve an existing safe policy.

  k: number of queries (in batch querying setting
  constrainHuman: a flag controls MR vs MR_k
  dry: no output to file if True
  rnd: random seed
  pf: only for Bayesian setting. ["prob that ith unknown feature is free" for i in range(self.numOfCons)]
    If None (by default), set randomly
  """
  mdp, consStates = officeNavigation(spec, gamma)

  numOfCons = len(consStates)
  consProbs = [pf + pfStep * random.random() for _ in range(numOfCons)]

  print 'consProbs', zip(range(numOfCons), consProbs)

  agent = ConsQueryAgent(mdp, consStates, consProbs=consProbs, constrainHuman=constrainHuman)

  # true free features, randomly generated
  trueFreeFeatures = filter(lambda idx: random.random() < consProbs[idx], range(numOfCons))
  # if require existence of safe policies after querying: setting relevant features of a dominating policy to be free features
  #relFeats, domPis = agent.findRelevantFeaturesAndDomPis()
  #trueFreeFeatures = agent.findViolatedConstraints(random.choice(domPis))
  # or hand designed
  print 'true free features', trueFreeFeatures

  if not agent.initialSafePolicyExists():
    # when the initial safe policy does not exist, we sequentially pose queries to find one safe policy
    print 'initial safe policy does not exist'

    methods = ['opt', 'iisAndRelpi', 'iisOnly', 'relpiOnly', 'maxProb', 'piHeu', 'random']
    #methods = ['iisAndRelpi', 'iisOnly', 'relpiOnly', 'maxProb', 'piHeu', 'random']
    queries = {}
    times = {}
    # these are assigned when ouralg is run
    iiss = None
    relFeats = None

    # keep track of algorithms' answers on whether problems are solvable
    answer = None
    thisAnswer = None

    for method in methods:
      print method
      queries[method] = []
      times[method] = []

      # ======== timed session ========
      start = time.time()

      if method == 'opt':
        agent = OptQueryForSafetyAgent(mdp, consStates, consProbs=consProbs)
      elif method == 'iisAndRelpi':
        agent = GreedyForSafetyAgent(mdp, consStates, consProbs=consProbs, useIIS=True, useRelPi=True)
        # record this to get an idea how difficult these tasks are
        # (iisAndRelpi compute both sets anyway, so record here)
        iiss = agent.iiss
        relFeats = agent.piRelFeats
      elif method == 'iisOnly':
        agent = GreedyForSafetyAgent(mdp, consStates, consProbs=consProbs, useIIS=True, useRelPi=False)
      elif method == 'relpiOnly':
        agent = GreedyForSafetyAgent(mdp, consStates, consProbs=consProbs, useIIS=False, useRelPi=True)
      elif method == 'maxProb':
        agent = MaxProbSafePolicyExistAgent(mdp, consStates, consProbs=consProbs)
      elif method == 'piHeu':
        agent = DomPiHeuForSafetyAgent(mdp, consStates, consProbs=consProbs)
      elif method == 'random':
        agent = DescendProbQueryForSafetyAgent(mdp, consStates, consProbs=consProbs)
      else:
        raise Exception('unknown method', method)

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

      # ======== timed session ends ========
      end = time.time()

      # the alg must return an answer
      assert thisAnswer != None

      # make sure all the algorithms give the same answer. otherwise imp error
      if answer == None:
        answer = thisAnswer
      else:
        assert answer == thisAnswer, {'other methods say': answer, method + ' says': thisAnswer}

      # make sure that, if safe policy exists, safe policy found
      if thisAnswer == EXIST:
        # may use other ways? most algorithms check this before returning anyway
        assert agent.safePolicyExist()

      times[method].append(end - start)

    print 'queries', queries
    print 'times', times
    print 'safe policy', answer

    if dry:
      print 'dry run. no output'
    else:
      lb = pf; ub = pf + pfStep
      # write to file
      pickle.dump({'q': queries, 't': times, 'iiss': iiss, 'relFeats': relFeats, 'solvable': answer == 'exist'},\
                  open(str(spec.width) + '_' + str(spec.height) + '_' + str(len(spec.carpets)) + '_' +\
                       str(lb) + '_' + str(ub) + '_' + str(rnd) + '.pkl', 'wb'))
  else:
    # when initial safe policies exist, we want to improve such a safe policy using batch queries
    print 'initial policy exists'

    agent = SafeImproveAgent(mdp, consStates, constrainHuman=constrainHuman)

    # we bookkeep the dominating policies for all domains. check whether if we have already computed them.
    # if so we do not need to compute them again.
    domainFileName = 'domain_' + str(numOfCarpets) + '_' + str(rnd) + '.pkl'
    if os.path.exists(domainFileName):
      data = pickle.load(open(domainFileName, 'rb'))
      if data == 'INITIALIZED':
        # failure in computing dom pi. do not try again.
        print "ABORT"
        return
      else:
        (relFeats, domPis, domPiTime) = data
    else:
      # don't save anything if we are dryrun
      if not dry:
        pickle.dump('INITIALIZED', open(domainFileName, 'wb'))

      # find dom pi (which may be used to find queries and will be used for evaluation)
      start = time.time()
      relFeats, domPis = agent.findRelevantFeaturesAndDomPis()
      end = time.time()
      domPiTime = end - start

      print "num of rel feats", len(relFeats)

      if not dry:
        pickle.dump((relFeats, domPis, domPiTime), open(domainFileName, 'wb'))

    methods = ['alg1', 'chain', 'naiveChain', 'relevantRandom', 'random', 'nq']

    # decide the true changeable features for expected regrets
    numpy.random.seed(2 * (1 + rnd)) # avoid weird coupling, e.g., the ones that are queried are exactly the true changeable ones
    if len(agent.allCons) < k:
      raise Exception('k is larger than the number of unknown features so no need to select queries. abort.')
    violableIndices = numpy.random.choice(range(len(agent.allCons)), k, replace=False)
    violableCons = [agent.allCons[_] for _ in violableIndices]

    for method in methods:
      start = time.time()
      if method == 'brute':
        q = agent.findMinimaxRegretConstraintQBruteForce(k, relFeats, domPis)
      elif method == 'reallyBrute':
        # really brute still need domPis to find out MR...
        q = agent.findMinimaxRegretConstraintQBruteForce(k, agent.allCons, domPis)
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

      if dry:
        print 'dry run. no output'
      else:
        saveToFileForSafePiImprove(method, k, numOfCarpets, constrainHuman, q, mrk, runTime, regret)

def saveToFileForSafePiImprove(method, k, numOfCarpets, constrainHuman, q, mrk, runTime, regret):
  ret = {}
  ret['mrk'] = mrk
  ret['regret'] = regret
  ret['time'] = runTime
  ret['q'] = q

  postfix = 'mrk' if constrainHuman else 'mr'

  # not distinguishing mr and mrk in filenames, so use a subdirectory
  pickle.dump(ret, open(method + '_' + postfix + '_' + str(k) + '_' + str(numOfCarpets) + '_' + str(rnd) + '.pkl', 'wb'))

def setRandomSeed(rnd):
  print 'random seed', rnd
  random.seed(rnd)
  numpy.random.seed(rnd)
  scipy.random.seed(rnd)

if __name__ == '__main__':
  # default values
  method = None
  k = 1
  constrainHuman = False
  dry = False # do not save to files if dry run

  numOfCarpets = 10
  numOfBoxes = 0
  size = 5

  rnd = 0 # set a dummy random seed if no -r argument

  pf = 0 # the prob. that a feature is free
  pfStep = 1

  batch = False # run batch experiments

  try:
    opts, args = getopt.getopt(sys.argv[1:], 's:k:n:cr:dp:b')
  except getopt.GetoptError:
    raise Exception('Unknown flag')
  for opt, arg in opts:
    if opt == '-k':
      k = int(arg)
    elif opt == '-s':
      size = int(arg)
    elif opt == '-n':
      numOfCarpets = int(arg)
    elif opt == '-c':
      constrainHuman = True
    elif opt == '-d':
      # disable dry run if output to file
      dry = True
    elif opt == '-p':
      # proportion of free features
      pf = float(arg)
    elif opt == '-b':
      batch = True
    elif opt == '-r':
      rnd = int(arg)
      setRandomSeed(rnd)
    else:
      raise Exception('unknown argument')

  if batch:
    # elements are (num of carpets, pf, pfStep)
    settingCandidates = [#([8, 9, 10, 11, 12], [0], 1),
                         ([10], [0, 0.2, 0.4, 0.6, 0.8], 0.2),
                         ([10], [0, 0.25, 0.5], 0.5),
                        ]

    for rnd in range(1000):
      for (carpetNums, pfRange, pfStep) in settingCandidates:
        for numOfCarpets in carpetNums:
          for pf in pfRange:
            # reset random seed in each iteration
            setRandomSeed(rnd)

            spec = squareWorld(size, numOfCarpets, avoidBorder=False)
            experiment(spec, k, constrainHuman, dry, rnd, pf, pfStep)
  else:
    #spec = carpetsAndWallsDomain()
    #spec = squareWorld(size, numOfCarpets, avoidBorder=False)
    #spec = squareWorld(size, numOfCarpets, avoidBorder=True)

    #spec = toySokobanWorld()
    spec = sokobanWorld()
    experiment(spec, k, constrainHuman, dry, rnd, pf=pf, pfStep=pfStep, gamma=0.9)
