import os
import pickle
import pylab
from numpy import mean
from matplotlib.ticker import MaxNLocator
from util import standardErr, createOrAppend

names = {'opt': 'Optimal',
         'myopic': 'Myopic', 'myopicReward': 'Myopic (Reward First)', 'myopicFeature': 'Myopic (Feature First)',
         'batch': 'Batch', 'dompi': 'Dom-Pi'}
markers = {'opt': 'r*-',
           'myopic': 'bv-', 'myopicReward': 'bv--', 'myopicFeature': 'bv:',
           'batch': 'bo-', 'dompi': 'g^-'}

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

  print xlabel, 'vs', ylabel
  fig = pylab.figure()
  ax = pylab.gca()
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

def histogram(x, xlabel, filename):
  fig = pylab.figure()

  pylab.hist(x)

  pylab.xlabel(xlabel)

  fig.savefig(filename + ".pdf", dpi=300, format="pdf")
  pylab.close()

def plotLegend():
  ax = pylab.gca()
  figLegend = pylab.figure(figsize=(4, 2.5))
  pylab.figlegend(*ax.get_legend_handles_labels(), loc='upper left')
  figLegend.savefig("legend.pdf", dpi=300, format="pdf")
  pylab.close()

def plotDifferenceOfTwoAlgs(x1, x2, xlabel, filename):
  diff = map(lambda elem1, elem2: elem1 - elem2, x1, x2)
  histogram(diff, xlabel, filename)

if __name__ == '__main__':
  font = {'size': 19}
  pylab.matplotlib.rc('font', **font)

  from config import trialsStart, trialsEnd, numsOfCarpets, numsOfSwitches, methods, costsOfQuery

  values = {}
  numOfQueries = {}
  returns = {}
  expectedReturns = {}
  times = {}

  for rnd in range(trialsStart, trialsEnd):
    filename = str(rnd) + '.pkl'
    # simply skip trials that are not run
    if os.path.exists(filename):
      results = pickle.load(open(filename, 'rb'))

      for numOfCarpets in numsOfCarpets:
        for numOfSwitches in numsOfSwitches:
          for costOfQuery in costsOfQuery:
            configKey = (numOfCarpets, numOfSwitches, costOfQuery)
            for method in methods:
              numQ = len(results[configKey][method]['queries'])

              createOrAppend(values, (configKey, method), results[configKey][method]['value'])
              createOrAppend(numOfQueries, (configKey, method), numQ)
              createOrAppend(returns, (configKey, method), results[configKey][method]['value'] - costOfQuery * numQ)
              createOrAppend(expectedReturns, (configKey, method), results[configKey][method]['expValue'] - costOfQuery * numQ)
              createOrAppend(times, (configKey, method), results[configKey][method]['time'])

  # plot different statistics in different figures
  statNames = ['objective', 'expected_obj', 'optimal_pi', 'number_of_queries', 'computation_time']
  statFuncs = [returns, expectedReturns, values, numOfQueries, times]

  # plot numOfCarpets as x-axis. plot different # of switches in different figures
  for numOfSwitches in numsOfSwitches:
    for costOfQuery in costsOfQuery:
      for (sName, sFunc) in zip(statNames, statFuncs):
        plot(numsOfCarpets, lambda method, numOfCarpets: sFunc[(numOfCarpets, numOfSwitches, costOfQuery), method], methods,
             xlabel='# of carpets', ylabel=sName,
             filename=sName + '_' + str(numOfSwitches) + '_' + str(costOfQuery), intXAxis=True)

  # to compare different algs
  for numOfCarpets in numsOfCarpets:
    for numOfSwitches in numsOfSwitches:
      for costOfQuery in costsOfQuery:
        configKey = (numOfCarpets, numOfSwitches, costOfQuery)

        batchResults = expectedReturns[configKey, 'batch']
        myopicResults = expectedReturns[configKey, 'myopic']
        domPiResults = expectedReturns[configKey, 'dompi']

        plotDifferenceOfTwoAlgs(batchResults, myopicResults, 'batch - myopic', 'batch_myopic_diff_'
                                + str(numOfCarpets) + '_' + str(numOfSwitches) + '_' + str(costOfQuery))
        plotDifferenceOfTwoAlgs(batchResults, domPiResults, 'batch - dompi', 'batch_domi_diff_'
                                + str(numOfCarpets) + '_' + str(numOfSwitches) + '_' + str(costOfQuery))
