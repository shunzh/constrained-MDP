VERBOSE = False
DEBUG = False

METHOD = 'gurobi'
#METHOD = 'cplex'

# experiment configuration
trialsStart = 0
trialsEnd = 1000
settingCandidates = [#([8, 9, 10, 11, 12], [0], 1),
                     ([14, 16], [0], 1),
                     #([10], map(lambda _: 0.1 * _, range(6)), 0.5),
                     #([14], map(lambda _: 0.1 * _, range(6)), 0.5)
                    ]

size = 6
walls = 5

methods = ['oracle',
           'opt',
           #'optLocked', 'optFree',
           'iisAndRelpi',
           #'iisOnly', 'relpiOnly',
           #'iisAndRelpi1',
           #'iisAndRelpi2',
           'iisAndRelpi3',
           #'iisAndRelpi4',
           'maxProb',
           #'maxProbF', 'maxProbIF', # variatiosn of maxProb
           'piHeu',
           #'setcoverWithValue', 'piHeuWithValue', # valuebased
           'random'
          ]
