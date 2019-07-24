VERBOSE = False
DEBUG = False

METHOD = 'gurobi'
#METHOD = 'cplex'

# experiment configuration
trials = 1000
settingCandidates = [([10, 15, 20], [0], 1),
                     #([10], map(lambda _: 0.1 * _, range(9)), 0.2),
                     ([10, 15, 20], map(lambda _: 0.1 * _, range(6)), 0.5),
                    ]

size = 6

methods = ['oracle',
           #'opt',
           #'optLocked', 'optFree',
           #'iisAndRelpi',
           #'iisOnly', 'relpiOnly',
           #'iisAndRelpi1',
           'iisAndRelpi2',
           #'iisOnly2', 'relpiOnly2',
           'maxProb',
           #'maxProbF', 'maxProbIF', # variatiosn of maxProb
           'piHeu',
           #'setcoverWithValue', 'piHeuWithValue', # valuebased
           'random']
