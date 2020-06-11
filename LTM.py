from scipy.spatial import distance
import numpy as np
import networkx as nx
import math

class LTM:
    def __init__(self, g: nx.Graph, seeds, visitors: dict, POIs):
        self.__g = g
        self.__seeds = seeds
        self.__weights = dict()
        self.__threshold = dict()
        self.__visitors = visitors
        self.__POIcount = POIs
        self.__fillThreshold()
        self.__fillWeights()
        self.__infectedNodes = self.__computeInfluence()

    def __fillThreshold(self):
        for i in range(self.__g.number_of_nodes()):
            self.__threshold[i] = 1 / self.__g.degree[i]

    def __fillWeights(self):
        for i in range(self.__g.number_of_nodes()):
            v1 = self.__visitors[i]
            v1keys = set(v1.keys())
            for n in self.__g.edges(i):
                neighbour = n[1]
                v2 = self.__visitors[neighbour]
                v2keys = set(v2.keys())
                intersection = list(v1keys.intersection(v2keys))
                vec2 = np.zeros(len(intersection))
                vec1 = np.zeros(len(intersection))
                for j in range(len(intersection)):
                    poi = intersection[j]
                    vec1[j] = v1[poi]
                    vec2[j] = v2[poi]
                if len(intersection) > 0:
                    print(vec1)
                    print(vec2)
                    weight = distance.cosine(vec1, vec2)
                else:
                    weight = 0
                self.__weights[i] = {neighbour: weight}
                self.__weights[neighbour] = {i: weight}

    def __computeInfluence(self):
        k = len(self.__seeds)
        totalIterationInfluenced = dict()
        currActiveNodes = self.__seeds
        totalInfluenced = self.__seeds
        while k > 0:
            totalIterationInfluenced = self.__computeCurrDiffusion(currActiveNodes)
            if totalInfluenced.size() == len(totalIterationInfluenced):
                break
            currActiveNodes = totalIterationInfluenced
            totalInfluenced = totalIterationInfluenced
            k = k-1
        return len(totalIterationInfluenced)

    def __computeCurrDiffusion(self, currActive: set):
        currActiveList = list(currActive)
        for activeVertex in currActiveList:
            size = len(self.__g.edges[activeVertex])
            for j in range(size):
                neighbour = self.__g.edges[activeVertex][j]
                if neighbour in currActiveList:
                    diff = self.__threshold[neighbour] - self.__weights[activeVertex][neighbour]
                    if diff <= 0:
                        currActiveList.append(neighbour)
                        self.__threshold[neighbour] = 0.0
        return set(currActiveList)

    def getInfectedNodes(self):
        return self.__infectedNodes
