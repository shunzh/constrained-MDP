VERBOSE = False
DEBUG = False

METHOD = 'gurobi'
#METHOD = 'cplex'

# experiment configuration
trialsStart = 0
trialsEnd = 1000

exp = 2

if exp == 1:
  settingCandidates = [([10, 12, 14], [5], [0], 1),
                       ([14], [5], map(lambda _: 0.1 * _, range(6)), 0.5),
                      ]

  size = 6

  earlyStop = None

  methods = [#'oracle',
             'opt',
             'iisAndRelpi',
             'iisOnly', 'relpiOnly',
             'iisAndRelpi3',
             #'maxProb',
             #'maxProbF', 'maxProbIF', # variations of maxProb
             #'piHeu',
             'random'
            ]

elif exp == 2:
  settingCandidates = [([40], [20], map(lambda _: 0.1 * _, range(6)), 0.5)
                      ]

  size = 10

  earlyStop = 5

  methods = ['oracle',
             #'opt',
             'iisAndRelpi',
             'iisAndRelpi3',
             'maxProb',
             'maxProbF', 'maxProbIF', # variations of maxProb
             'piHeu',
            ]
else:
  raise Exception('unknown exp')
