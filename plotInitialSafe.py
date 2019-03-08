import pickle

import matplotlib
import pylab
from matplotlib.ticker import FormatStrFormatter
from numpy import mean

import util
from util import standardErr

rndSeeds = 1000

width = height = 5

lensOfQ = {}
times = {}

#proportionRange = [0.01] + [0.1 * proportionInt for proportionInt in range(10)] + [0.99]
carpetNums = [8, 9, 10, 11, 12]

# will check what methods are run from data
includeOpt = True
includeRandom = False

methods = (['opt'] if includeOpt else []) \
          + ['iisAndRelpi', 'iisOnly', 'relpiOnly', 'maxProb', 'piHeu'] \
          + (['random'] if includeRandom else [])

markers = {'opt': 'r*-', 'iisAndRelpi': 'bo-', 'iisOnly': 'bs--', 'relpiOnly': 'bd-.', 'maxProb': 'g^-', 'piHeu': 'm+-', 'random': 'c.-'}
names = {'opt': 'Optimal', 'iisAndRelpi': 'SetCover', 'iisOnly': 'SetCover (IIS)', 'relpiOnly': 'SetCover (rel. feat.)', 'maxProb': 'Greed. Prob.',\
         'piHeu': 'Most-Likely', 'random': 'Descending'}

def addFreq(elem, counter): counter[elem] += 1

# output the diffierence of two vectors
vectorDiff = lambda v1, v2: map(lambda e1, e2: e1 - e2, v1, v2)

# for output as latex table
outputFormat = lambda d: '$' + str(round(mean(d), 4)) + ' \pm ' + str(round(standardErr(d), 4)) + '$'

def plot(x, y, yci, methods, xlabel, ylabel, filename):
  """
  general script for plotting using pylab
  """
  fig = pylab.figure()

  ax = pylab.gca()
  for method in methods:
    print method, y(method), yci(method)
    lines = ax.errorbar(x, y(method), yci(method), fmt=markers[method], mfc='none', label=names[method], markersize=10, capsize=5)

  pylab.xlabel(xlabel)
  pylab.ylabel(ylabel)
  pylab.legend()
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

  fig.savefig(filename + ".pdf", dpi=300, format="pdf")
  
  pylab.close()


def plotNumVsProportion(pfRange, pfStep):
  """
  Plot the the number of queried features vs the proportion of free features
  """
  # fixed carpet num for this exp
  carpets = 10

  for method in methods:
    for pf in pfRange:
      lensOfQ[method, pf] = []
      times[method, pf] = []

  validInstances = []

  for rnd in range(rndSeeds):
    # set to true if this instance is valid (no safe init policy)
    rndProcessed = False

    for pf in pfRange:
      try:
        pfUb = pf + pfStep

        filename = str(width) + '_' + str(height) + '_' + str(carpets) + '_' + str(pf) + '_' + str(pfUb) + '_' + str(rnd) + '.pkl'
        data = pickle.load(open(filename, 'rb'))
      except IOError:
        print filename, 'not exist'
        continue

      # number of features queried
      for method in methods:
        lensOfQ[method, pf].append(len(data['q'][method]))
        times[method, pf].append(data['t'][method])
      
      if not rndProcessed:
        rndProcessed = True

        validInstances.append(rnd)
    
  print 'valid instances', len(validInstances)
  assert len(validInstances) > 0

  # show cases where method1 and method2 are different with proportion. for further debugging methods
  diffInstances = lambda pf, method1, method2:\
                  (pf, method1, method2,\
                  filter(lambda _: _[1] != _[2], zip(validInstances, lensOfQ[method1, pf], lensOfQ[method2, pf])))

  """
  for pf in pfRange: 
    print diffInstances(pf, 'iisAndRelpi', 'iisOnly')
  """

  # plot figure
  x = pfRange
  y = lambda method: [mean(vectorDiff(lensOfQ[method, pf], lensOfQ[methods[0], pf])) for pf in pfRange]
  yci = lambda method: [standardErr(vectorDiff(lensOfQ[method, pf], lensOfQ[methods[0], pf])) for pf in pfRange]

  plot(x, y, yci, methods, '$p_f$', '# of Queried Features (' + names[methods[0]] + ' as baseline)', 'lensOfQPf' + str(int(pfStep * 10)))


def plotNumVsCarpets():
  """
  plot the num of queried features / computation time vs. num of carpets
  """
  for method in methods:
    for carpetNum in carpetNums:
      lensOfQ[method, carpetNum] = []
      times[method, carpetNum] = []

  iisSizes = {}
  iisSizesVec = {}

  domPiSizes = {}
  domPiSizesVec = {}

  solveableIns = {}
  validInstances = {}
  for carpetNum in carpetNums:
    iisSizes[carpetNum] = util.Counter()
    iisSizesVec[carpetNum] = []

    domPiSizes[carpetNum] = util.Counter()
    domPiSizesVec[carpetNum] = []

    solveableIns[carpetNum] = util.Counter()

    validInstances[carpetNum] = []

  for rnd in range(rndSeeds):
    for carpetNum in carpetNums:
      try:
        filename = str(width) + '_' + str(height) + '_' + str(carpetNum) + '_0_1_' +  str(rnd) + '.pkl'
        data = pickle.load(open(filename, 'rb'))
      except IOError:
        print filename, 'not exist'
        continue

      # number of features queried
      for method in methods:
        lensOfQ[method, carpetNum].append(len(data['q'][method]))
        times[method, carpetNum].append(data['t'][method])

      validInstances[carpetNum].append(rnd)

      # print the case where ouralg is suboptimal
      if 'opt' in methods and len(data['q']['opt']) < len(data['q']['iisAndRelpi']):
        print 'rnd', rnd, 'carpetNum', carpetNum, 'opt', data['q']['opt'], 'iisAndRelpi', data['q']['iisAndRelpi']

      addFreq(len(data['iiss']), iisSizes[carpetNum])
      iisSizesVec[carpetNum].append(len(data['iiss']))

      addFreq(len(data['relFeats']), domPiSizes[carpetNum])
      domPiSizesVec[carpetNum].append(len(data['relFeats']))

      addFreq(data['solvable'], solveableIns[carpetNum])


  print 'iiss', [round(mean(iisSizesVec[carpetNum]), 2) for carpetNum in carpetNums]
  print 'relFeats', [round(mean(domPiSizesVec[carpetNum]), 2) for carpetNum in carpetNums]

  print 'validins', [len(validInstances[carpetNum]) for carpetNum in carpetNums]
  print 'solvable', [round(1.0 * solveableIns[carpetNum][True] / len(validInstances[carpetNum]), 2) for carpetNum in carpetNums]

  print '# of queries'
  x = carpetNums
  # use the first method as baseline, a bit hacky here.
  y = lambda method: [mean(vectorDiff(lensOfQ[method, carpetNum], lensOfQ[methods[0], carpetNum])) for carpetNum in carpetNums]
  yci = lambda method: [standardErr(vectorDiff(lensOfQ[method, carpetNum], lensOfQ[methods[0], carpetNum])) for carpetNum in carpetNums]
  plot(x, y, yci, methods, '# of Carpets', '# of Queried Features (' + names[methods[0]] + ' as baseline)', 'lensOfQCarpets')

  print 'compute time'
  x = carpetNums
  y = lambda method: [mean(times[method, carpetNum]) for carpetNum in carpetNums]
  yci = lambda method: [standardErr(times[method, carpetNum]) for carpetNum in carpetNums]
  plot(x, y, yci, methods, '# of Carpets', 'Computation Time (sec.)', 'timesCarpets')

font = {'size': 13}
matplotlib.rc('font', **font)

pfCandidates = [(0.2, [0, 0.2, 0.4, 0.6, 0.8]),\
#                (0.3, [0, 0.35, 0.7]),\
                (0.5, [0, 0.25, 0.5])]

# exp 1: varying num of carpets
#plotNumVsCarpets()

# exp 2: varying pfs
for (pfStep, pfRange) in pfCandidates: plotNumVsProportion(pfRange, pfStep)
