def findHighestFrequencyElement(feats, sets, weight=lambda _: 1):
  """
  Here we want to use elements to cover sets.
  This function finds the element that appears in the most number of the sets.

  sets: [[elements in one sets] for all sets]
  weights: find the element with the maximum weighted frequency. unweighted by default
  """
  if len(sets) == 0: return None

  appearenceFreq = {}
  
  for e in feats:
    appearenceFreq[e] = weight(e) * sum(e in s for s in sets)
  
  # return the index of the element that has the most appearances
  return max(appearenceFreq.iteritems(), key=lambda _: _[1])[0]
  
def coverFeat(feat, sets):
  """
  Find the new set of sets if feat is covered.
  We only need to remove the sets that contain feat.
  """
  return filter(lambda s: feat not in s, sets)

def removeFeat(feat, sets):
  """
  Find the new set of sets if feat is removed.
  We remove feat, and remove sets that are reducible (which are supersets of any other set).
  """
  newSets = map(lambda s: tuple(set(s) - {feat}), sets)
  newSets = list(set(newSets)) # kill duplicates
  newSets = filter(lambda s: not any(set(otherSet).issubset(s) for otherSet in newSets if otherSet != s), newSets)
  return map(lambda s: tuple(s), newSets)

def killSupersets(sets):
  """
  A hacky way to remove sets that are supersets of others.
  
  {{1}, {1, 2}} --> {{1}}
  """
  return removeFeat(None, sets)

def leastNumElemSets(feat, sets):
  """
  Find the smallest set that contains feat and return the size when feat is removed.
  """
  setsWithoutFeat = map(lambda s: set(s) - {feat}, sets)
  minSizedSet = min(setsWithoutFeat, key=lambda s: len(s))
  return minSizedSet

def elementExists(feat, sets):
  return any(feat in s for s in sets)

def oshimai(sets):
  return len(sets) == 0 or any(len(s) == 0 for s in sets)

"""
DEPRECATED look at the dual form of the set, not in this way..
"""
def findElementThatRemovesMostSets(feats, sets, admissibleProbs):
  """
  We focus on how many sets can be removed: either by covered or by removing an element.
  """
  # either no more sets to cover, or any iis becomes empty
  if len(sets) == 0 or any(len(s) == 0 for s in sets): return None

  expectNumRemainingSets = {}

  for e in feats:
    expectNumRemainingSets[e] = admissibleProbs[e] * len(coverFeat(e, sets)) + (1 - admissibleProbs[e]) * len(removeFeat(e, sets))
    
  return min(expectNumRemainingSets.iteritems(), key=lambda _: _[1])[0]
