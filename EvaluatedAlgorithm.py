# -*- coding: utf-8 -*-

from RecommenderMetrics import RecommenderMetrics
from BuildDataSet import BuildDataSet


class Eval_Algo_Proc:

    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name

    def Evaluate(self, dataset, evaluateTopN, n=10, verbose=True):
        metrics = {}
        self.algorithm.fit(dataset.GetTrainSet())
        predictions = self.algorithm.test(dataset.GetTestSet())
        metrics["RMSE"] = RecommenderMetrics.RMSE(predictions)
        metrics["MAE"] = RecommenderMetrics.MAE(predictions)

        if evaluateTopN:
            self.algorithm.fit(dataset.GetLOOCVTrainSet())
            leftOneOut_prediction = self.algorithm.test(
                dataset.GetLOOCVTestSet())

            allPredictions = self.algorithm.test(
                dataset.GetLOOCVAntiTestSet())

            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)

            metrics["HR"] = RecommenderMetrics.HitRate(
                topNPredicted, leftOneOut_prediction)

            metrics["cHR"] = RecommenderMetrics.CumulativeHitRate(
                topNPredicted, leftOneOut_prediction)

            metrics["ARHR"] = RecommenderMetrics.AverageReciprocalHitRank(
                topNPredicted, leftOneOut_prediction)

            self.algorithm.fit(dataset.GetFullTrainSet())
            allPredictions = self.algorithm.test(
                dataset.GetFullAntiTestSet())
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)

            metrics["Coverage"] = RecommenderMetrics.UserCoverage(topNPredicted,
                                                                  dataset.GetFullTrainSet().n_users,
                                                                  ratingThreshold=4.0)

            metrics["Diversity"] = RecommenderMetrics.Diversity(
                topNPredicted, dataset.GetSimilarities())

            metrics["Novelty"] = RecommenderMetrics.Novelty(topNPredicted,
                                                            dataset.GetPopularityRankings())

        return metrics

    def GetName(self):
        return self.name

    def GetAlgorithm(self):
        return self.algorithm
