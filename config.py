VERBOSE = False
DEBUG = False

OPT_METHOD = 'gurobi'
#OPT_METHOD = 'cplex'

earlyStop = None

# experiment configuration
trialsStart = 0
trialsEnd = 1000

methods = ['opt', 'myopic', 'dompi', 'dompiUniform']
