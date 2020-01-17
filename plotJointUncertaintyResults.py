import os
import pickle

import pylab
from numpy import mean
from matplotlib.ticker import MaxNLocator
from util import standardErr

names = {'myopic': 'Myopic', 'batch': 'Batch', 'dompi': 'Dom-Pi'}
markers = {'myopic': 'bv-', 'batch': 'bo-', 'dompi': 'g^-'}

def plot(x, y, methods, xlabel, ylabel, filename, intXAxis=False, intYAxis=False, xAxis=None, ylim=None):
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
    ax.errorbar(xAxis, yMean(method), yCI(method), fmt=markers[method], mfc='none', label=names[method],
                markersize=15, capsize=10,  linewidth=2)

  pylab.xlabel(xlabel)
  pylab.ylabel(ylabel)

  if intXAxis:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  if intYAxis:
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
  if ylim is not None:
    pylab.ylim(ylim)

  pylab.gcf().subplots_adjust(bottom=0.15, left=0.15)
  fig.savefig(filename + ".pdf", dpi=300, format="pdf")

  plotLegend()

  pylab.close()

def plotLegend():
  ax = pylab.gca()
  figLegend = pylab.figure(figsize=(3.5, 4.2))
  pylab.figlegend(*ax.get_legend_handles_labels(), loc='upper left')
  figLegend.savefig("legend.pdf", dpi=300, format="pdf")

def parseJointUncertaintyResults():
  from config import trialsStart, trialsEnd, numsOfCarpets, methods, costOfQuery

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
          for numOfCarpets in numsOfCarpets:
            numQ = len(results[method][numOfCarpets]['queries'])

            values[method].append(results[method][numOfCarpets]['value'])
            numOfQueries[method].append(numQ)
            returns[method].append(results[method][numOfCarpets]['value'] - costOfQuery * numQ)
            times[method].append(results[method][numOfCarpets]['time'])

  statNames = ['objective value', 'value of safely-optimal $\pi$', 'number of queries', 'computation time (sec.)']
  statFuncs = [returns, values, numOfQueries, times]

  for (sName, sFunc) in zip(statNames, statFuncs):
    plot(numsOfCarpets, sFunc, methods, '# of carpets', sName, sName)

if __name__ == '__main__':
    parseJointUncertaintyResults()
