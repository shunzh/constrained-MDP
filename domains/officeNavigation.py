import random

import domainConstructors

# constants for objects in the environment
import util

R = 'R'
_ = '_'
W = 'W'
C = 'C'
S = 'S'
D = 'D'
B = 'B'

# some consts
OPEN = 1
CLOSED = 0

STEPPED = 1
CLEAN = 0

ON = 1
OFF = 0 

OPENDOOR = 'openDoor'
CLOSEDOOR = 'closeDoor'
TURNOFFSWITCH = 'turnOffSwitch'

TERMINATE = 'terminate'


class Spec():
  """
  A object that describes the specifications of a tabular environment with walls, doors, boxes, carpets, etc.
  """
  def __init__(self, width, height, robot, switches, walls, doors, boxes, carpets, horizon=None):
    self.width = width
    self.height = height
    
    self.robot = robot
    self.switches = switches
    
    self.walls = walls
    self.doors = doors
    self.boxes = boxes
    self.carpets = carpets
    
    self.horizon = horizon


def toyWorldConstructor(map, horizon=None):
  """
  Create a Spec object based on what is plotted in map

  :param map: a 2-d array
  :param horizon: horizon of the time. None of infinite
  :return: a Spec obj
  """
  # just make plotting easier
  width = len(map[0])
  height = len(map)

  robot = None
  switches = []
  walls = []
  doors = []
  carpets = []
  boxes = []

  for i in range(height):
    for j in range(width):
      if type(map[i][j]) == tuple:
        elems = map[i][j]
      else:
        elems = (map[i][j],)

      for elem in elems:
        if elem == R:
          robot = (j, i)
        elif elem == S:
          switches.append((j, i))
        elif elem == W:
          walls.append((j, i))
        elif elem == D:
          doors.append((j, i))
        elif elem == C:
          carpets.append((j, i))
        elif elem == B:
          boxes.append((j, i))

  if robot == None: raise Exception('Robot location not specified!')
  if len(switches) == 0: raise Exception('Should specify at least one switch')

  return Spec(width, height, robot, switches, walls, doors, boxes, carpets, horizon)

"""
A list of toy domains.
"""
def carpetsAndWallsDomain():
  """
  some proof of concepts domains
  the optimal query policy should be clear in these domains
  """
  map0 = [[R, S],
          [_, S],
          [_, _]]

  map1 = [[_, C, S],
          [_, W, W],
          [R, C, S],
          [_, W, W],
          [_, C, S]]

  map2 = [[R, C, C, S],
          [_, W, W, W],
          [_, C, C, S]]

  map3 = [[_, C, S, S, S],
          [R, W, W, W, W],
          [_, C, S, S, S]]

  return toyWorldConstructor(map0)

# some toy domains for need-to-be-reverted features (boxes)
def toySokobanWorld():
  # the robot should go straight ahead to turn off the switch (while pushing the box away)
  # then detour to the other side of the box to push it back
  map = [[_, _, _, _, _],
         [R, B, S, _, _]]
  return toyWorldConstructor(map, horizon=10)

def sokobanWorld():
  # when only one the box is reversible, the robot should push both boxes away and detour them to reach the switch
  # and then push the reversible boxes back. the performance is the same regardless which box needs to be reverted.
  # when both boxes need to be reverted, the performance is the worst since the robot needs to push box 1 back before
  # reaching the switch.
  map = [[_, W, _, _, _, W, W, _, _, _],
         [R, B, _, _, _, B, _, _, _, S]]
  return toyWorldConstructor(map, horizon=25)


"""
A list of parameterized domains.
These are randomly generated rather than hand-specified.
"""
# parameterized worlds
def squareWorld(size, numOfCarpets, numOfWalls, numOfSwitches=1):
  """
  Squared world with width = height = size.
  The robot and the switch are at opposite corners (unless randomSwitch==True).
  Carpets and walls are uniformly randomly generated.
  No doors.
  """
  if type(size) is tuple:
    width = size[0]
    height = size[1]
  else:
    width = height = size
  
  robot = (0, 0)

  # these objects are absent
  doors = []
  boxes = [] # no need to put in boxes for now

  # leave a safe path to reach the exit state
  possibleLocs = [(x, y) for x in range(width) for y in range(height)]
  possibleLocs.remove(robot)

  # generate the list of the locations of all objects, make sure they don't overlap
  objectLocs = util.sampleSubset(possibleLocs, numOfWalls + numOfCarpets + numOfSwitches)
  walls = objectLocs[:numOfWalls]
  carpets = objectLocs[numOfWalls:numOfWalls + numOfCarpets]
  switches = objectLocs[numOfWalls + numOfCarpets:]
  assert len(switches) == numOfSwitches # make sure indexing is correct

  return Spec(width, height, robot, switches, walls, doors, boxes, carpets)

def squareWorldStats(spec):
  """
  :param spec: a square world instance
  :return: some stats that see if they correlate with some empirical results
  """
  switchDis = 1.0 * sum(util.getManhattanDistance(switch1, switch2) for switch1 in spec.switches for switch2 in spec.switches if switch2 != switch1)\
              / (len(spec.switches) * (len(spec.switches) - 1))

  switchToRobotDis = 1.0 * sum(util.getManhattanDistance(spec.robot, switch) for switch in spec.switches)\
                     / len(spec.switches)

  return {'switchDis': switchDis, 'switchToRobotDis': switchToRobotDis}

def parameterizedSokobanWorld(size, numOfBoxes):
  width = height = size
  
  robot = (0, 0)
  switches = [(width - 1, height - 1)]
  
  walls = []
  doors = []
  
  boxes = [random.choice([(x, y) for x in range(width) for y in range(height) if x != 0 or y != 0]) for _ in range(numOfBoxes)]
  carpets = [] # no non-reversible features
  
  horizon = size * 2
  
  return Spec(width, height, robot, switches, walls, doors, boxes, carpets, horizon)

def plotDomain(spec):
  # check whether the world looks as expected
  for y in range(spec.height):
    print '%3d' % y,
    for x in range(spec.width):
      if (x, y) in spec.walls:
        print '[ W]',
      elif (x, y) in spec.boxes:
        print '[ B]',
      elif (x, y) in spec.switches:
        print '[S%1d]' % spec.switches.index((x, y)),
      elif (x, y) == spec.robot:
        print '[ R]',
      elif spec.carpets.count((x, y)) == 1:
        print '[C%1d]' % spec.carpets.index((x, y)),
      elif spec.carpets.count((x, y)) > 1:
        print '[%2d*' % spec.carpets.index((x, y)),
      else:
        print '[  ]',
    print
  print '  ',
  for x in range(spec.width):
    print '%4d' % x,
  print

def officeNavigationTask(spec, rewardProbs, gamma=.9):
  """
  spec: specification of the factored mdp
  gamma: discounting factor
  """
  # robot's location
  locIndex = 0

  # door indices
  dIndexStart = locIndex + 1
  dSize = len(spec.doors)
  dIndices = range(dIndexStart, dIndexStart + dSize)

  # box indices
  bIndexStart = dIndexStart + dSize
  bSize = len(spec.boxes)
  bIndices = range(bIndexStart, bIndexStart + bSize)

  # switch indices
  sIndexStart = bIndexStart + bSize
  sSize = len(spec.switches)
  sIndices = range(sIndexStart, sIndexStart + sSize)

  # time index (may or may not be used)
  # time is needed when there are horizon-dependent constraints
  tIndex = sIndexStart + sSize

  moveActs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
  directionalActs = moveActs + [(0, 0)]
  aSets = directionalActs + [TURNOFFSWITCH]

  plotDomain(spec)

  def boxMovable(idx, s, a):
    """
    A helper function to decide whether the robot can move a box in a location
    return True if the box represented by s[idx] can be moved with a applied in state s
    False otherwise
    """
    assert a in directionalActs

    box = s[idx]
    boxP = (box[0] + a[0], box[1] + a[1]) # box prime, the next location without considering constraints
    # box is not moved across the border and not into walls or other boxes
    if boxP[0] >= 0 and boxP[0] < spec.width and boxP[1] >= 0 and boxP[1] < spec.height\
       and not boxP in spec.walls\
       and not boxP in [s[bIndex] for bIndex in bIndices]:
      return True
    else:
      return False

  """
  factored transition functions
  """
  def navigate(s, a):
    loc = s[locIndex]
    if a in directionalActs:
      sp = (loc[0] + a[0], loc[1] + a[1])
      # not blocked by borders, closed doors
      # not pushing towards a non-movable box
      # not blocked by walls
      if (sp[0] >= 0 and sp[0] < spec.width and sp[1] >= 0 and sp[1] < spec.height) and\
         not any(s[idx] == CLOSED and sp == spec.doors[idx - dIndexStart] for idx in dIndices) and\
         not any(sp == s[idx] and not boxMovable(idx, s, a) for idx in bIndices) and\
         not sp in spec.walls:
        return sp
    # otherwise the agent is blocked and return the unchanged location
    return loc
  
  def doorOpGen(idx, door):
    def doorOp(s, a):
      loc = s[locIndex]
      doorState = s[idx]
      if a in [OPENDOOR, CLOSEDOOR]:
        if loc in [(door[0] - 1, door[1]), (door[0], door[1])]:
          if a == CLOSEDOOR: doorState = CLOSED
          elif a == OPENDOOR: doorState = OPEN
          # otherwise the door state is unchanged
      return doorState
    return doorOp
 
  def boxOpGen(idx):
    def boxOp(s, a):
      loc = s[locIndex]
      box = s[idx]
      if a in directionalActs and navigate(s, a) == box:
        if boxMovable(idx, s, a):
          newBox = (box[0] + a[0], box[1] + a[1])
          return newBox
      # otherwise the box state is unchanged
      return box
    return boxOp

  def switchOpGen(idx, switch):
    def switchOp(s, a):
      loc = s[locIndex]
      switchState = s[idx]
      if loc == switch and a == TURNOFFSWITCH:
        return OFF
      else:
        return switchState # unchaged
    return switchOp

  def timeElapse(s, a):
    return s[tIndex] + 1
  
  # all physically possible locations
  allVisitableLocations = [(x, y) for x in range(spec.width) for y in range(spec.height) if (x, y) not in spec.walls]

  # cross product of possible values of all features
  # location, door1, door2, carpets, switch
	# time is added if horizon is not None (the task is time dependent)
  sSets = [allVisitableLocations] +\
          [[CLOSED, OPEN] for _ in spec.doors] +\
          [allVisitableLocations for _ in spec.boxes] +\
          [[OFF, ON]] +\
          ([range(spec.horizon + 1)] if spec.horizon != None else [])

  # the transition function is also factored
  # the i-th element is a function defined on S x A -> S_i
  tFunc = [navigate] +\
          [doorOpGen(i, spec.doors[i - dIndexStart]) for i in dIndices] +\
          [boxOpGen(i) for i in bIndices] +\
          [switchOpGen(i, spec.switches[i - sIndexStart]) for i in sIndices] +\
          ([timeElapse] if spec.horizon != None else [])

  s0List = [spec.robot] +\
           [CLOSED for _ in spec.doors] +\
           spec.boxes +\
           [ON for _ in spec.switches] +\
           ([0] if spec.horizon != None else [])

  s0 = tuple(s0List)
  print 'init state', s0

  # terminal conditions
  if spec.horizon != None:
    terminal = lambda s: s[tIndex] == spec.horizon
  else:
    # let the episode end when any switch is off
    terminal = lambda s: any(s[sIndex] == OFF for sIndex in sIndices)
    # define a terminal state
    #terminal = lambda s: s[locIndex] == (spec.width - 1, spec.height - 1)

  # use random rewards?
  #rewards = [1 + random.random() for _ in spec.switches]

  # give reward of 2 when the target switch is turned off
  # otherwise, give reward of random.random(), stored in randomRewardDict
  # note that the robot does not have the action to turn the switch back on
  def rewardFuncGen(switchIndex):
    def rFunc(s, a):
      loc = s[locIndex]
      # if the robot turns off a switch
      if loc == spec.switches[switchIndex - sIndexStart] and s[switchIndex] == ON and a == TURNOFFSWITCH:
        return 1
      else:
        return 0
    return rFunc

  # create the list of reward candidates, in the form of [(reward_func, prob)]
  rFunc = [(rewardFuncGen(sIdx), rewardProbs[sIdx - sIndexStart]) for sIdx in sIndices]

  mdp = domainConstructors.DeterministicFactoredMDP(sSets, aSets, rFunc, tFunc, s0, gamma, terminal)

  # implement the set of constrained states based on feature representation
  # consStates is [[states that violate the i-th constraint] for i in all constraints]
  # Note that implementation here does not distinguish free and need-to-be-reverted features
  # since we implement them both as constraints in linear programming anyway.

  # carpets are locked features by default
  carpetCons = [[s for s in mdp.S if s[locIndex] == _] for _ in spec.carpets]
  # boxes are need-to-be-reverted features by default
  boxCons = [[s for s in mdp.S if terminal(s) and s[bIdx] != s0[bIdx]] for bIdx in bIndices]
  consStates = carpetCons + boxCons

  # goal states are that the switch needs to be turned off in the end
  #goalStates = [s for s in mdp.S for sIndex in sIndices if s[sIndex] == OFF]
  goalStates = []

  return mdp, consStates, goalStates
