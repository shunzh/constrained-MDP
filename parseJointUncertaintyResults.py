import os
import pickle
from numpy import mean
from util import standardErr


def parseJointUncertaintyResults():
  from config import trialsStart, trialsEnd, methods, costOfQuery

  constructStatsDict = lambda: {method: [] for method in methods}

  values = constructStatsDict()
  numOfQueries = constructStatsDict()
  returns = constructStatsDict()
  times = constructStatsDict()

  for rnd in range(trialsStart, trialsEnd):
    filename = str(rnd) + '.pkl'
    if os.path.exists(filename):
      results = pickle.load(open(filename, 'rb'))

      for method in methods:
        if method in results.keys():
          numQ = len(results[method]['queries'])

          values[method].append(results[method]['value'])
          numOfQueries[method].append(numQ)
          returns[method].append(results[method]['value'] - costOfQuery * numQ)
          times[method].append(results[method]['time'])

      if returns['myopic'][-1] < returns['opt'][-1]: print rnd

  statNames = ['objective value', 'value of safely-optimal $\pi$', 'number of queries', 'computation time (sec.)']
  statFuncs = [returns, values, numOfQueries, times]

  print methods
  for (statName, statFunc) in zip(statNames, statFuncs):
    print statName
    for method in methods:
      print '& %.3f $\\pm$ %.3f' % (mean(statFunc[method]), standardErr(statFunc[method]))
    print '\\\\'

if __name__ == '__main__':
    parseJointUncertaintyResults()
