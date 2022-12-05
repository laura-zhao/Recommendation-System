# -*- coding: utf-8 -*-

from BuildDataSet import BuildDataSet
from Eval_Algo_Proc import Eval_Algo_Proc


class Evaluator:

    algorithms = []

    def __init__(self, dataset, rankings):
        ed = BuildDataSet(dataset, rankings)
        self.dataset = ed

    def AddAlgorithm(self, algorithm, name):
        alg = Eval_Algo_Proc(algorithm, name)
        self.algorithms.append(alg)

    def Evaluate(self, evaluateTopN):
        evaluated_res = {}
        for algorithm in self.algorithms:
            evaluated_res[algorithm.GetName()] = algorithm.Evaluate(
                self.dataset, evaluateTopN)

        print("\n")

        if evaluateTopN:
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                "Algorithm", "RMSE", "MAE", "HR", "cHR", "ARHR", "Coverage", "Diversity", "Novelty"))
            for (name, metrics) in evaluated_res.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                    name, metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["cHR"], metrics["ARHR"],
                    metrics["Coverage"], metrics["Diversity"], metrics["Novelty"]))
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for (name, metrics) in evaluated_res.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(
                    name, metrics["RMSE"], metrics["MAE"]))

    def SampleTopNRecs(self, ml, testSubject=85, k=10):
        for algo in self.algorithms:
            print("\nUsing recommendation algorithm", algo.GetName())
            trainSet = self.dataset.GetFullTrainSet()
            algo.GetAlgorithm().fit(trainSet)
            testSet = self.dataset.GetAntiTestSetForUser(testSubject)
            predictions = algo.GetAlgorithm().test(testSet)
            recommendations = []
            print("\nWe recommend:")
            for userID, movieID, actualRating, estimatedRating, _ in predictions:
                intMovieID = int(movieID)
                recommendations.append((intMovieID, estimatedRating))

            recommendations.sort(key=lambda x: x[1], reverse=True)

            for ratings in recommendations[:10]:
                print(ml.getMovieName(ratings[0]), ratings[1])
