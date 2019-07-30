import collections
import pickle

import matplotlib
import pylab
from matplotlib.ticker import MaxNLocator
from numpy import mean

from util import standardErr

from config import trials, size

# FIXME assuming squared domain for now
width = height = size

from config import methods
# don't plot random, out of range
if 'random' in methods: methods.remove('random')
print methods

#baseline = 'opt'
baseline = 'oracle'

markers = {'oracle': 'r*--',
           'opt': 'r*-',
           'iisAndRelpi': 'bo-', 'iisAndRelpi1': 'bs-', 'iisAndRelpi2': 'bd-',
           'iisOnly': 'bo--', 'relpiOnly': 'bo-.',
           'iisAndRelpi3': 'bv-',
           'iisOnly3': 'bv--', 'relpiOnly3': 'bv-.',
           'maxProb': 'g^-', 'maxProbF': 'g^--', 'maxProbIF': 'g^-.',
           'piHeu': 'm+-', 'random': 'c.-',
           'setcoverWithValue': 'bo-', 'piHeuWithValue': 'm+-'}
names = {'oracle': 'Oracle',
         'opt': 'Optimal',
         'iisAndRelpi': '$h_{SC}$',
         'iisOnly': '$h_{SC}$ (IIS)', 'relpiOnly': '$h_{SC}$ (rel. feat.)',
         'iisAndRelpi1': '$h_{SC}$ w/ P[T]',
         'iisAndRelpi2': '$h_{CR}$ max',
         'iisAndRelpi3': '$h_{CR}$ sum',
         'iisOnly3': '$h_{CR}$ (IIS)', 'relpiOnly3': '$h_{CR}$ (rel. feat.)',
         'maxProb': 'Greed. Prob.', 'maxProbF': 'Greed. Prob. Feasible', 'maxProbIF': 'Greed. Prob. Infeasible',
         'piHeu': 'Most-Likely', 'random': 'Descending',
         'setcoverWithValue': 'Weighted Set Cover', 'piHeuWithValue': 'Most-Likely with Value'}

# output the difference of two vectors
vectorDiff = lambda v1, v2: map(lambda e1, e2: e1 - e2, v1, v2)
# output the ratio of two vectors
vectorDivide = lambda v1, v2: map(lambda e1, e2: 1.0 * e1 / e2, v1, v2)


def plot(x, y, methods, xlabel, ylabel, filename, integerAxis=False, xAxis=None):
  """
  plot data.

  :param x: x axis
  :param y: y(method, x_elem) is a vector that contains raw data
  :param methods: methods to plot, each has a legend
  :param xlabel: name of xlabel
  :param ylabel: name of ylabel
  :param filename: output to filename.pdf
  :return:
  """
  if xAxis is None: xAxis = x

  yMean = lambda method: [mean(y(method, xElem)) for xElem in x]
  yCI = lambda method: [standardErr(y(method, xElem)) for xElem in x]

  fig = pylab.figure()

  ax = pylab.gca()
  print xlabel, ylabel
  for method in methods:
    print method, yMean(method), yCI(method)
    ax.errorbar(xAxis, yMean(method), yCI(method), fmt=markers[method], mfc='none', label=names[method], markersize=10, capsize=5)

  pylab.xlabel(xlabel)
  pylab.ylabel(ylabel)

  if integerAxis:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

  fig.savefig(filename + ".pdf", dpi=300, format="pdf")

  plotLegend()

  pylab.close()

def printTex(head, data):
  print head,
  for d in data: print ' & ', d,
  print '\\\\'

def plotLegend():
  ax = pylab.gca()
  figLegend = pylab.figure(figsize=(3.2, 2))
  pylab.figlegend(*ax.get_legend_handles_labels(), loc='upper left')
  figLegend.savefig("legend.pdf", dpi=300, format="pdf")

def plotMeanOfRatioWrtBaseline(x, y, methods, xlabel, ylabel, filename, integerAxis=False, xAxis=None):
  """
  plot data with a specified baseline.

  mean (value of this method / value of the baseline)
  """
  # can set displayed x to be different
  if xAxis is None: xAxis = x

  yMean = lambda method: [mean(vectorDivide(y(method, xElem), y(baseline, xElem))) for xElem in x]
  yCI = lambda method: [standardErr(vectorDivide(y(method, xElem), y(baseline, xElem))) for xElem in x]

  fig = pylab.figure()

  ax = pylab.gca()
  for method in methods:
    #print method, yMean(method), yCI(method)
    ax.errorbar(xAxis, yMean(method), yCI(method),
                fmt=markers[method], mfc='none', label=names[method], markersize=10, capsize=5)

  pylab.xlabel(xlabel)
  pylab.ylabel(ylabel)

  if integerAxis:
    # x-axis should be integers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

  fig.savefig(filename + ".pdf", dpi=300, format="pdf")

  plotLegend() # make sure legend is plotted somewhere

  pylab.close()

def plotRatioOfMeanDiffWrtBaseline(x, y, methods, xlabel, ylabel, filename, integerAxis=False):
  """
  plot data with a specified baseline.

  mean values of this method / mean values of the baseline
  """
  yMean = lambda method: [mean(y(method, xElem)) for xElem in x]
  yCI = lambda method: [standardErr(vectorDiff(y(method, xElem), y(baseline, xElem))) for xElem in x]

  fig = pylab.figure()

  ax = pylab.gca()
  for method in methods:
    #print method, yMean(method), yCI(method)
    ax.errorbar(x, vectorDivide(yMean(method), yMean(baseline)), vectorDivide(yCI(method), yMean(baseline)),
                fmt=markers[method], mfc='none', label=names[method], markersize=10, capsize=5)

  pylab.xlabel(xlabel)
  pylab.ylabel(ylabel)

  if integerAxis:
    # x-axis should be integers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

  fig.savefig(filename + ".pdf", dpi=300, format="pdf")

  pylab.close()

def scatter(x, y, xlabel, ylabel, filename):
  fig = pylab.figure()

  for method in methods:
    # weirdly scatter doesn't have a fmt parameter. setting marker and color separately
    pylab.scatter(x, y(method), c=markers[method][0], marker=markers[method][1])

  ax = pylab.gca()
  pylab.xlabel(xlabel)
  pylab.ylabel(ylabel)
  pylab.legend()

  fig.savefig(filename + ".pdf", dpi=300, format="pdf")

  pylab.close()

def plotNumVsProportion(carpetNum, pfRange, pfStep):
  """
  Plot the the number of queried features vs the proportion of free features
  """
  lensOfQ = {}
  times = {}

  # fixed carpet num for this exp
  for method in methods:
    for pf in pfRange:
      lensOfQ[method, pf] = []
      times[method, pf] = []

  validInstances = []

  for rnd in range(trials):
    # set to true if this instance is valid (no safe init policy)
    rndProcessed = False

    for pf in pfRange:
      try:
        pfUb = pf + pfStep

        filename = str(width) + '_' + str(height) + '_' + str(carpetNum) + '_' + str(pf) + '_' + str(pfUb) + '_' + str(rnd) + '.pkl'
        data = pickle.load(open(filename, 'rb'))
      except IOError:
        #print filename, 'not exist'
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

  # to identify the runs where two methods have different outputs
  diffInstances = lambda pf, method1, method2:\
                  (pf, method1, method2,
                  filter(lambda _: _[1] != _[2], zip(validInstances, lensOfQ[method1, pf], lensOfQ[method2, pf])))

  # plot figure
  x = pfRange
  # plot the midpoint of intervals
  xAxis = map(lambda _: _ + pfStep / 2, pfRange)

  y = lambda method, pf: lensOfQ[method, pf]
  plot(x, y, methods, '$p_f$', '# of Queried Features',
       'lensOfQPf' + str(carpetNum) + '_' + str(pfStep),
       xAxis=xAxis)
  plotMeanOfRatioWrtBaseline(x, y, methods, '$p_f$', '# of Queried Features / Optimal',
                             'lensOfQPf' + str(carpetNum) + '_' + str(pfStep) + '_meanOfRatio',
                             xAxis=xAxis)

  y = lambda method, pf: times[method, pf]
  plot(x, y, methods, '$p_f$', 'Computation Time (sec.)', 'timesPf', xAxis=xAxis)

def plotNumVsCarpets(carpetNums):
  """
  plot the num of queried features / computation time vs. num of carpets
  plotting data with pf = 0, pfStep = 1
  """
  lensOfQ = {}
  lensOfQRelPhi = {}
  safePiValues = {}
  times = {}

  for method in methods:
    for carpetNum in carpetNums:
      lensOfQ[method, carpetNum] = []
      times[method, carpetNum] = []
      safePiValues[method, carpetNum] = []

    for relFeatNum in range(max(carpetNums) + 1):
      # relevant features is going to be at most the number of unknown features anyway
      # num of queries asked using method when relFeatNum
      lensOfQRelPhi[method, relFeatNum] = []

  iiss = {}
  domPis = {}

  # a trial is valid (and its data should be counted) when no initial safe policy exists
  validInstances = {}
  # trials where the robot is able to find a safe policy after querying
  # (instead of claiming that ho safe policies exist)
  solvableIns = {}

  # initialize dictionaries
  for carpetNum in carpetNums:
    iiss[carpetNum] = []
    domPis[carpetNum] = []
    solvableIns[carpetNum] = []
    validInstances[carpetNum] = []

  for rnd in range(trials):
    for carpetNum in carpetNums:
      try:
        filename = str(width) + '_' + str(height) + '_' + str(carpetNum) + '_0_1_' +  str(rnd) + '.pkl'
        data = pickle.load(open(filename, 'rb'))
      except IOError:
        #print filename, 'not exist'
        continue

      # see which features appear in relevant features of any dominating policy
      relFeats = len(filter(lambda _: any(_ in relFeats for relFeats in data['relFeats']), range(carpetNum)))
      # get stats
      for method in methods:
        lensOfQ[method, carpetNum].append(len(data['q'][method]))
        lensOfQRelPhi[method, relFeats].append(len(data['q'][method]))
        if 'valuesOfSafePis' in data.keys() and len(data['valuesOfSafePis']) > 0:
          safePiValues[method, carpetNum].append(data['valuesOfSafePis'][method])
        times[method, carpetNum].append(data['t'][method])

      iiss[carpetNum].append(len(data['iiss']))
      domPis[carpetNum].append(len(data['relFeats']))
      # num of relevant features

      validInstances[carpetNum].append(rnd)
      if data['solvable']: solvableIns[carpetNum].append(rnd)

      """
      # print the case where ouralg is suboptimal for analysis
      if 'opt' in methods and len(data['q']['opt']) < len(data['q']['iisAndRelpi']):
        print 'rnd', rnd, 'carpetNum', carpetNum, 'opt', data['q']['opt'], 'iisAndRelpi', data['q']['iisAndRelpi']
      """

  printTex('\\# of trials w/ no initial safe policies',
           [len(validInstances[carpetNum]) for carpetNum in carpetNums])
  printTex('proportion of trials where safe policies exist',
           [round(1.0 * len(solvableIns[carpetNum]) / len(validInstances[carpetNum]), 2) for carpetNum in carpetNums])
  printTex('average \\# of IISs',
           [round(mean(iiss[carpetNum]), 2) for carpetNum in carpetNums])
  printTex('average \\# of dominating policies',
           [round(mean(domPis[carpetNum]), 2) for carpetNum in carpetNums])

  print '# of queries'
  x = carpetNums
  # absolute number of queried features
  y = lambda method, carpetNum: lensOfQ[method, carpetNum]
  plot(x, y, methods, '# of Carpets', '# of Queried Features', 'lensOfQCarpets')
  plotMeanOfRatioWrtBaseline(x, y, methods, '# of Carpets', '# of Queried Features / Optimal',
                             'lensOfQCarpets_meanOfRatio', integerAxis=True)

  #y = lambda method, carpetNum: safePiValues[method, carpetNum]
  #plot(x, y, methods, '# of Carpets', 'Values of Safe Policy', 'safePiValuesCarpets')

  y = lambda method, carpetNum: times[method, carpetNum]
  plot(x, y, methods, '# of Carpets', 'Computation Time (sec.)', 'timesCarpets')

  # plot num of features queried based on the num of dom pis
  x = range(max(carpetNums))
  y = lambda method, relFeat: lensOfQRelPhi[method, relFeat]

  plotMeanOfRatioWrtBaseline(x, y, methods, '# of Relevant Features', '# of Queried Features / Optimal',
                             'lensOfQCarpets_rel_meanOfRatio', integerAxis=True)

if __name__ == '__main__':
  font = {'size': 13}
  matplotlib.rc('font', **font)

  from config import settingCandidates

  for (carpetNums, pfRange, pfStep) in settingCandidates:
    if len(carpetNums) > 1 and len(pfRange) == 1:
      # exp 1: varying num of carpets
      plotNumVsCarpets(carpetNums)
    else:
      for carpetNum in carpetNums:
        plotNumVsProportion(carpetNum, pfRange, pfStep)

