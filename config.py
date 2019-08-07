VERBOSE = False
DEBUG = False

METHOD = 'gurobi'
#METHOD = 'cplex'

# experiment configuration
trialsStart = 0
trialsEnd = 600
settingCandidates = [#([8, 9, 10, 11, 12], [0], 1),
                     ([14, 16, 18], [0], 1),
                     #([10], map(lambda _: 0.1 * _, range(6)), 0.5),
                     #([14], map(lambda _: 0.1 * _, range(6)), 0.5),
                    ]

size = 6

methods = ['oracle',
           #'opt',
           #'optLocked', 'optFree',
           'iisAndRelpi',
           #'iisOnly', 'relpiOnly',
           #'iisAndRelpi1',
           #'iisAndRelpi2',
           'iisAndRelpi3',
           #'iisOnly3', 'relpiOnly3',
           'iisAndRelpi4',
           'maxProb',
           #'maxProbF', 'maxProbIF', # variatiosn of maxProb
           'piHeu',
           #'setcoverWithValue', 'piHeuWithValue', # valuebased
           'random']
