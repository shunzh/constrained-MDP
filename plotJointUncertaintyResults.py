import os
import pickle
import pylab
from numpy import mean
from matplotlib.ticker import MaxNLocator

from domains.officeNavigation import squareWorldStats
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
  numOfMatch = sum(_ == 0 for _ in x)

  fig = pylab.figure()

  pylab.hist(x, bins=map((0.2).__mul__, range(-5, 6)))
  # draw a vertical line that highlight the number of matches
  pylab.plot((0, 0), (0, numOfMatch), marker='+', color='black', linewidth=3, markeredgewidth=2, markersize=15)

  pylab.xlabel(xlabel)
  pylab.ylabel('frequency')

  pylab.gcf().subplots_adjust(bottom=0.15, left=0.15)
  fig.savefig(filename + ".pdf", dpi=300, format="pdf")
  pylab.close()

def correlation(x, y, xlabel, ylabel, filename):
  fig = pylab.figure()
  pylab.scatter(x, y)

  pylab.xlabel(xlabel)
  pylab.ylabel(ylabel)

  fig.savefig(filename + ".pdf", dpi=300, format="pdf")
  pylab.close()

def plotLegend():
  ax = pylab.gca()
  figLegend = pylab.figure(figsize=(4, 2.5))
  pylab.figlegend(*ax.get_legend_handles_labels(), loc='upper left')
  figLegend.savefig("legend.pdf", dpi=300, format="pdf")
  pylab.close()


if __name__ == '__main__':
  font = {'size': 19}
  pylab.matplotlib.rc('font', **font)

  from config import trialsStart, trialsEnd, numsOfCarpets, numsOfSwitches, methods, costsOfQuery

  values = {}
  numOfQueries = {}
  returns = {}
  expectedReturns = {}
  times = {}

  switchDis = {}
  switchToRobotDis = {}
  domPiNums = {}

  for rnd in range(trialsStart, trialsEnd):
    filename = str(rnd) + '.pkl'
    # simply skip trials that are not run
    if os.path.exists(filename):
      results = pickle.load(open(filename, 'rb'))

      for numOfCarpets in numsOfCarpets:
        for numOfSwitches in numsOfSwitches:
          spec, domPiNum = results[(numOfCarpets, numOfSwitches)]
          stats = squareWorldStats(spec)
          createOrAppend(switchDis, (numOfCarpets, numOfSwitches), stats['switchDis'])
          createOrAppend(switchToRobotDis, (numOfCarpets, numOfSwitches), stats['switchToRobotDis'])
          createOrAppend(domPiNums, (numOfCarpets, numOfSwitches), domPiNum)

          for costOfQuery in costsOfQuery:
            configKey = (numOfCarpets, numOfSwitches, costOfQuery)
            for method in methods:
              numQs = map(lambda _: len(_), results[configKey][method]['queries'])

              createOrAppend(values, (configKey, method), mean(results[configKey][method]['value']))
              createOrAppend(numOfQueries, (configKey, method), mean(numQs))
              createOrAppend(returns, (configKey, method), mean(results[configKey][method]['value']) - costOfQuery * mean(numQs))
              createOrAppend(times, (configKey, method), mean(results[configKey][method]['time']))

  # plot different statistics in different figures
  statNames = ['objective', 'policy value', 'number of queries', 'computation time (sec.)']
  statFuncs = [returns, values, numOfQueries, times]

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

        for comparedHeuristic in ['myopic', 'dompi']:
          batchResults = returns[configKey, 'batch']
          comparedResults = returns[configKey, comparedHeuristic]

          batchDiff = [e1 - e2 for e1, e2 in zip(batchResults, comparedResults)]

          histogram(batchDiff, 'batch - ' + comparedHeuristic,
                    'batch_' + comparedHeuristic + '_diff_' + str(numOfCarpets) + '_' + str(numOfSwitches) + '_' + str(costOfQuery))

          correlation(switchDis[numOfCarpets, numOfSwitches], batchDiff, 'average switch distances', 'batch - ' + comparedHeuristic,
                      'batch_' + comparedHeuristic + '_switchDis_' + str(numOfCarpets) + '_' + str(numOfSwitches) + '_' + str(costOfQuery))
          correlation(switchToRobotDis[numOfCarpets, numOfSwitches], batchDiff, 'switch to robot distances', 'batch - ' + comparedHeuristic,
                      'batch_' + comparedHeuristic + '_switchToRobotDis_' + str(numOfCarpets) + '_' + str(numOfSwitches) + '_' + str(costOfQuery))
          correlation(domPiNums[numOfCarpets, numOfSwitches], batchDiff, '# of dominating policies', 'batch - ' + comparedHeuristic,
                      'batch_' + comparedHeuristic + '_dompis_' + str(numOfCarpets) + '_' + str(numOfSwitches) + '_' + str(costOfQuery))
