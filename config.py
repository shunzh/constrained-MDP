VERBOSE = False
DEBUG = False

METHOD = 'gurobi'
#METHOD = 'cplex'

# experiment configuration
trialsStart = 0
trialsEnd = 1000
settingCandidates = [([10, 12, 14], [0, 10], [0], 1),
                     #([16], [10], map(lambda _: 0.1 * _, range(6)), 0.5),
                    ]

size = 6

methods = ['oracle',
           'opt',
           #'optLocked', 'optFree',
           'iisAndRelpi',
           #'iisOnly', 'relpiOnly',
           #'iisAndRelpi1',
           #'iisAndRelpi2',
           'iisAndRelpi3',
           'maxProb',
           #'maxProbF', 'maxProbIF', # variations of maxProb
           'piHeu',
           #'setcoverWithValue', 'piHeuWithValue', # valuebased
           #'random'
          ]
