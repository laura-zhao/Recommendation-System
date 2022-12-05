# -*- coding: utf-8 -*-

from MovieLens import MovieLens
from surprise import SVD
from surprise import NormalPredictor
from Evaluator import Evaluator
from surprise.model_selection import GridSearchCV

import random
import numpy as np


def LoadMovieLensData():
    ml = MovieLens()
    data = ml.loadMovieLensLatestSmall()
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)


np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, dataset, rankings) = LoadMovieLensData()
param_grid = {'n_epochs': [20, 30], 'lr_all': [0.005, 0.010],
              'n_factors': [50, 100]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(dataset)

# best RMSE score
print("Best RMSE score attained: ", gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])


evaluator = Evaluator(dataset, rankings)

params = gs.best_params['rmse']
SVDtuned = SVD(n_epochs=params['n_epochs'],
               lr_all=params['lr_all'], n_factors=params['n_factors'])
evaluator.AddAlgorithm(SVDtuned, "SVD - Tuned")

SVDUntuned = SVD()
evaluator.AddAlgorithm(SVDUntuned, "SVD - Untuned")

evaluator.Evaluate(False)

evaluator.SampleTopNRecs(ml)
