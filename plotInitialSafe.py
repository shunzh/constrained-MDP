import collections
import pickle

import matplotlib
import pylab
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from numpy import mean

from util import standardErr

rndSeeds = 600

width = height = 5

lensOfQ = {}
lensOfQRelPhi = {}
times = {}

#carpetNums = [8, 9, 10, 11, 12]
carpetNums = [10, 11, 12]

includeOpt = False # if opt is run
includeRandom = False # random may be out of scope

#methods = (['opt'] if includeOpt else []) \
#          + ['iisAndRelpi', 'iisOnly', 'relpiOnly', 'maxProb', 'piHeu'] \
#          + (['random'] if includeRandom else [])
methods = ['setcoverWithValue', 'piHeuWithValue', 'random']
markers = {'opt': 'r*-', 'iisAndRelpi': 'bo-', 'iisOnly': 'bs--', 'relpiOnly': 'bd-.', 'maxProb': 'g^-', 'piHeu': 'm+-', 'random': 'c.-',
           'setcoverWithValue': 'bo-', 'piHeuWithValue': 'm+-'}
names = {'opt': 'Optimal', 'iisAndRelpi': 'SetCoverQuery', 'iisOnly': 'SetCoverQuery (IIS)', 'relpiOnly': 'SetCoverQuery (rel. feat.)',
         'maxProb': 'Greed. Prob.', 'piHeu': 'Most-Likely', 'random': 'Descending',
         'setcoverWithValue': 'Weighted Set Cover', 'piHeuWithValue': 'Most-Likely with Value'}

# output the difference of two vectors
vectorDiff = lambda v1, v2: map(lambda e1, e2: e1 - e2, v1, v2)
# output the ratio of two vectors. 1 if e2 == 0
vectorDivide = lambda v1, v2: map(lambda e1, e2: e1 / e2, v1, v2)


def plot(x, y, methods, xlabel, ylabel, filename):
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
  yMean = lambda method: [mean(y(method, xElem)) for xElem in x]
  yCI = lambda method: [standardErr(y(method, xElem)) for xElem in x]

  fig = pylab.figure()

  ax = pylab.gca()
  for method in methods:
    #print method, yMean(method), yCI(method)
    ax.errorbar(x, yMean(method), yCI(method), fmt=markers[method], mfc='none', label=names[method], markersize=10, capsize=5)

  pylab.xlabel(xlabel)
  pylab.ylabel(ylabel)

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

def plotMeanOfRatioWrtBaseline(x, y, methods, baseline, xlabel, ylabel, filename):
  """
  plot data with a specified baseline.

  mean (value of this method / value of the baseline)
  """
  yMean = lambda method: [mean(vectorDivide(y(method, xElem), y(baseline, xElem))) for xElem in x]
  yCI = lambda method: [standardErr(vectorDivide(y(method, xElem), y(baseline, xElem))) for xElem in x]

  fig = pylab.figure()

  ax = pylab.gca()
  for method in methods:
    #print method, yMean(method), yCI(method)
    ax.errorbar(x, yMean(method), yCI(method),
                fmt=markers[method], mfc='none', label=names[method], markersize=10, capsize=5)

  pylab.xlabel(xlabel)
  pylab.ylabel(ylabel)

  fig.savefig(filename + ".pdf", dpi=300, format="pdf")

  plotLegend() # make sure legend is plotted somewhere

  pylab.close()

def plotRatioOfMeanDiffWrtBaseline(x, y, methods, baseline, xlabel, ylabel, filename):
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
  y = lambda method, pf: lensOfQ[method, pf]
  plotRatioOfMeanDiffWrtBaseline(x, y, methods, 'opt', '$p_f$', '# of Queried Features / Optimal', 'lensOfQPf' + str(int(pfStep * 10)) + '_ratioOfMean')
  plotMeanOfRatioWrtBaseline(x, y, methods, 'opt', '$p_f$', '# of Queried Features / Optimal', 'lensOfQPf' + str(int(pfStep * 10)) + '_meanOfRatio')

def plotNumVsCarpets():
  """
  plot the num of queried features / computation time vs. num of carpets
  """
  for method in methods:
    for carpetNum in carpetNums:
      lensOfQ[method, carpetNum] = []
      times[method, carpetNum] = []

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

  for rnd in range(rndSeeds):
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
        times[method, carpetNum].append(data['t'][method])

      iiss[carpetNum].append(len(data['iiss']))
      domPis[carpetNum].append(len(data['relFeats']))
      # num of relevant features

      validInstances[carpetNum].append(rnd)
      if data['solvable']: solvableIns[carpetNum].append(rnd)

      # print the case where ouralg is suboptimal for analysis
      if 'opt' in methods and len(data['q']['opt']) < len(data['q']['iisAndRelpi']):
        print 'rnd', rnd, 'carpetNum', carpetNum, 'opt', data['q']['opt'], 'iisAndRelpi', data['q']['iisAndRelpi']

  printTex('\\# of trials w/ no initial safe policies',
           [len(validInstances[carpetNum]) for carpetNum in carpetNums])
  printTex('proportion of trials where safe policies exist',
           [round(1.0 * len(solvableIns[carpetNum]) / len(validInstances[carpetNum]), 2) for carpetNum in carpetNums])
  printTex('average \\# of IISs',
           [round(mean(iiss[carpetNum]), 2) for carpetNum in carpetNums])
  printTex('average \# of dominating policies',
           [round(mean(domPis[carpetNum]), 2) for carpetNum in carpetNums])

  print '# of queries'
  x = carpetNums
  # absolute number of queried features
  y = lambda method, carpetNum: lensOfQ[method, carpetNum]
  plot(x, y, methods, '# of Carpets', '# of Queried Features', 'lensOfQCarpets')
  #plotRatioOfMeanDiffWrtBaseline(x, y, methods, 'opt', '# of Carpets', '# of Queried Features / Optimal', 'lensOfQCarpets_ratioOfMean')
  #plotMeanOfRatioWrtBaseline(x, y, methods, 'opt', '# of Carpets', '# of Queried Features / Optimal', 'lensOfQCarpets_meanOfRatio')

  x = carpetNums
  y = lambda method, carpetNum: times[method, carpetNum]
  plot(x, y, methods, '# of Carpets', 'Computation Time (sec.)', 'timesCarpets')

  # plot num of features queried based on the num of dom pis
  x = range(max(carpetNums))
  y = lambda method, relFeat: lensOfQRelPhi[method, relFeat]

  #plotRatioOfMeanDiffWrtBaseline(x, y, methods, 'opt', '# of Relevant Features', '# of Queried Features / Optimal', 'lensOfQCarpets_rel_ratioOfMean')
  #plotMeanOfRatioWrtBaseline(x, y, methods, 'opt', '# of Relevant Features', '# of Queried Features / Optimal', 'lensOfQCarpets_rel_meanOfRatio')

if __name__ == '__main__':
  font = {'size': 13}
  matplotlib.rc('font', **font)

  pfCandidates = [(0.2, [0, 0.2, 0.4, 0.6, 0.8]),
                  #(0.3, [0, 0.35, 0.7]),
                  (0.5, [0, 0.25, 0.5])
                 ]

  # exp 1: varying num of carpets
  plotNumVsCarpets()

  # exp 2: varying pfs
  #for (pfStep, pfRange) in pfCandidates: plotNumVsProportion(pfRange, pfStep)
