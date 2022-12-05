# -*- coding: utf-8 -*-

from MovieLens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter

# using the 60th user as the prediction user
userForTest = '60'
n = 10

ml = MovieLens()
data = ml.loadMovieLensLatestSmall()

trainSet = data.build_full_trainset()

sim_options = {'name': 'cosine',
               'user_based': False
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

testUserInnerID = trainSet.to_inner_uid(userForTest)

# Get the top N items the user rated
testUserRatings = trainSet.ur[testUserInnerID]
kNeighbors = heapq.nlargest(n, testUserRatings, key=lambda t: t[1])

# Get similar items to top N items (weighted by rating)
candidates = defaultdict(float)
for itemID, rating in kNeighbors:
    similarityRow = simsMatrix[itemID]
    for innerID, score in enumerate(similarityRow):
        candidates[innerID] += score * (rating / 5.0)

# Build a dictionary of stuff the user has already seen
seen = {}
for itemID, rating in trainSet.ur[testUserInnerID]:
    seen[itemID] = 1

# Get top-rated items from similar users:
pos = 0
for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not itemID in seen:
        movieID = trainSet.to_raw_iid(itemID)
        print(ml.getMovieName(int(movieID)), ratingSum)
        pos += 1
        if (pos > 10):
            break
