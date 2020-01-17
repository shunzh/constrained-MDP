VERBOSE = False
DEBUG = False

OPT_METHOD = 'gurobi'
#OPT_METHOD = 'cplex'

# make this smaller because we need to find dom pis for 2^|\R| times in joint uncertainty works
earlyStop = 0.1

costOfQuery = 0.1

# experiment configuration
trialsStart = 0
trialsEnd = 1000

methods = ['myopic', 'batch', 'dompi']

numsOfCarpets = [10, 12, 14]
