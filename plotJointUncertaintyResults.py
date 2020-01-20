import os
import pickle
import pylab
from numpy import mean
from matplotlib.ticker import MaxNLocator
from util import standardErr, createOrAppend

names = {'opt': 'Optimal', 'myopic': 'Myopic', 'batch': 'Batch', 'dompi': 'Dom-Pi'}
markers = {'opt': 'r*-', 'myopic': 'bv-', 'batch': 'bo-', 'dompi': 'g^-'}

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

def plotLegend():
  ax = pylab.gca()
  figLegend = pylab.figure(figsize=(3.5, 4.2))
  pylab.figlegend(*ax.get_legend_handles_labels(), loc='upper left')
  figLegend.savefig("legend.pdf", dpi=300, format="pdf")

if __name__ == '__main__':
  from config import trialsStart, trialsEnd, numsOfCarpets, numsOfSwitches, methods, costOfQuery

  constructStatsDict = lambda: {(method, numOfCarpets): [] for method in methods for numOfCarpets in numsOfCarpets}

  values = constructStatsDict()
  numOfQueries = constructStatsDict()
  returns = constructStatsDict()
  times = constructStatsDict()

  for rnd in range(trialsStart, trialsEnd):
    filename = str(rnd) + '.pkl'
    # simply skip trials that are not run
    if os.path.exists(filename):
      results = pickle.load(open(filename, 'rb'))

      for numOfCarpets in numsOfCarpets:
        for numOfSwitches in numsOfSwitches:
          configKey = (numOfCarpets, numOfSwitches)
          for method in methods:
            numQ = len(results[configKey][method]['queries'])

            createOrAppend(values, (configKey, method), results[configKey][method]['value'])
            createOrAppend(numOfQueries, (configKey, method), numQ)
            createOrAppend(returns, (configKey, method), results[configKey][method]['value'] - costOfQuery * numQ)
            createOrAppend(times, (configKey, method), results[configKey][method]['time'])

  # plot different statistics in different figures
  statNames = ['objective', 'optimal_pi', 'number_of_queries', 'computation_time']
  statFuncs = [returns, values, numOfQueries, times]

  # just to be consistent with previous plotting, plot numOfCarpets as x-axis. plot different # of switches in different figures
  for numOfSwitches in numsOfSwitches:
    for (sName, sFunc) in zip(statNames, statFuncs):
      plot(numsOfCarpets, lambda method, numOfCarpets: sFunc[(numOfCarpets, numOfSwitches), method], methods,
           xlabel='# of carpets', ylabel=sName,
           filename=sName + '_' + str(numOfSwitches), intXAxis=True)
