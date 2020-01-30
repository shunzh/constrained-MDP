VERBOSE = False
DEBUG = False

OPT_METHOD = 'gurobi'
#OPT_METHOD = 'cplex'

# make this smaller because we need to find dom pis for 2^|\R| times in joint uncertainty works
earlyStop = 1

# for each domain configuration, sample the true reward fucntion and the true free features
sampleInstances = 10

# experiment configuration
trialsStart = 0
trialsEnd = 500

methods = ['myopic', 'batch', 'dompi']

numsOfCarpets = [10, 12, 14]
numsOfSwitches = [2, 4]
costsOfQuery = [0.01, 0.1, 0.2]
