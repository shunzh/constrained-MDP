import os
import pickle
from numpy import mean
from util import standardErr


def parseJointUncertaintyResults():
  from config import trialsStart, trialsEnd, methods

  costOfQ = 0.01

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
          values[method].append(results[method]['value'])
          numOfQueries[method].append(results[method]['numOfQueries'])
          returns[method].append(results[method]['value'] - costOfQ * results[method]['numOfQueries'])
          times[method].append(results[method]['time'])

  statNames = ['objective value', 'value of safely-optimal $\pi$', 'number of queries', 'computation time (sec.)']
  statFuncs = [returns, values, numOfQueries, times]

  for (statName, statFunc) in zip(statNames, statFuncs):
    print statName
    for method in methods:
      print '& %.2f $\\pm$ %.2f' % (mean(statFunc[method]), standardErr(statFunc[method]))
    print '\\\\'

if __name__ == '__main__':
    parseJointUncertaintyResults()
