# -*- coding: utf-8 -*-

from MovieLens import MovieLens
from surprise import SVD
from surprise import NormalPredictor
from Evaluator import Evaluator

import random
import numpy as np


def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (data, rankings)


np.random.seed(0)
random.seed(0)


(evaluationData, rankings) = LoadMovieLensData()
evaluator = Evaluator(evaluationData, rankings)

# Evaluate SVD Algorithm
SVDAlgorithm = SVD(random_state=10)
evaluator.AddAlgorithm(SVDAlgorithm, "SVD")

# Evaluate User-based KNN
UserKNN = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
evaluator.AddAlgorithm(UserKNN, "User KNN")

# Evaluate Item-based KNN
ItemKNN = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
evaluator.AddAlgorithm(ItemKNN, "Item KNN")

# SVD
SVD = SVD()
evaluator.AddAlgorithm(SVD, "SVD")

# SVD++
SVDPlusPlus = SVDpp()
evaluator.AddAlgorithm(SVDPlusPlus, "SVD++")

# RBM
RBM = RBMAlgorithm(epochs=20)
evaluator.AddAlgorithm(RBM, "RBM")

evaluator.Evaluate(True)
