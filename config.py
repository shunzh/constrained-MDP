DEBUG = False
VERBOSE = False

METHOD = 'gurobi'
#METHOD = 'cplex'

# experiment configuration
trials = 1000
settingCandidates = [([8, 9, 10, 11, 12], [0], 1),
                     ([10], [0, 0.2, 0.4, 0.6, 0.8], 0.2),
                     ([10], [0, 0.25, 0.5], 0.5),
                    ]

methods = ['opt', 'iisAndRelpi', 'iisOnly', 'relpiOnly', 'maxProb', 'maxProbF', 'maxProbIF', 'piHeu', 'random']
#methods = ['opt', 'iisAndRelpi', 'iisOnly', 'relpiOnly', 'maxProb', 'piHeu', 'random']
#methods = ['setcoverWithValue', 'piHeuWithValue', 'random']
