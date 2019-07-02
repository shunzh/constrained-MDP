DEBUG = False
VERBOSE = False

METHOD = 'gurobi'
#METHOD = 'cplex'

# experiment configuration
trials = 1000
settingCandidates = [([8, 9, 10, 11, 12], [0], 1),
                     #([10], map(lambda _: 0.1 * _, range(9)), 0.2),
                     #([10], [0, 0.25, 0.5], 0.5),
                    ]

methods = ['opt',
           'iisAndRelpi',
           #'iisOnly', 'relpiOnly',
           'iisAndRelpiOne',
           'maxProb',
           #'maxProbF', 'maxProbIF', # variatiosn of maxProb
           'piHeu',
           #'setcoverWithValue', 'piHeuWithValue', # valuebased
           'random']
