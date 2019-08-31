VERBOSE = False
DEBUG = False

METHOD = 'gurobi'
#METHOD = 'cplex'

# experiment configuration
trialsStart = 0
trialsEnd = 1000
settingCandidates = [#([10, 12, 14], [5], [0], 1),
                     ([15, 20, 25], [20], [0], 1),
                     #([14], [5], map(lambda _: 0.1 * _, range(6)), 0.5),
                    ]

size = 10
#size = 6

earlyStop = 20
#earlyStop = None

methods = ['oracle',
           #'opt',
           #'optLocked', 'optFree',
           'iisAndRelpi',
           #'iisOnly', 'relpiOnly',
           #'iisAndRelpi1',
           #'iisAndRelpi2',
           'iisAndRelpi3',
           'maxProb',
           'maxProbF', 'maxProbIF', # variations of maxProb
           'piHeu',
           #'setcoverWithValue', 'piHeuWithValue', # valuebased
           #'random'
          ]
