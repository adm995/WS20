from scipy.spatial import distance
import numpy as np
import networkx as nx
import math
from Similarities import Similarities


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
            if i in self.__visitors.keys():
                v1 = self.__visitors[i]
                for edge in self.__g.edges(i):
                    neighbour = edge[1]
                    if neighbour in self.__visitors.keys():
                        v2 = self.__visitors[neighbour]
                        weight = Similarities.get_cosine_similarity(v1, v2)
                        self.__weights[i] = {neighbour: weight}
                        self.__weights[neighbour] = {i: weight}

    def __computeInfluence(self):
        k = len(self.__seeds)
        totalIterationInfluenced = dict()
        currActiveNodes = self.__seeds
        totalInfluenced = self.__seeds
        while k > 0:
            totalIterationInfluenced = self.__computeCurrDiffusion(currActiveNodes)
            if len(totalInfluenced) == len(totalIterationInfluenced):
                break
            currActiveNodes = totalIterationInfluenced
            totalInfluenced = totalIterationInfluenced
            k = k-1
        return len(totalIterationInfluenced)

    def __computeCurrDiffusion(self, currActive: set):
        currActiveList = list(currActive)
        for activeVertex in currActiveList:
            for neighbour in self.__g.edges(activeVertex):
                if neighbour in currActiveList:
                    diff = self.__threshold[neighbour] - self.__weights[activeVertex][neighbour]
                    if diff <= 0:
                        currActiveList.append(neighbour)
                        self.__threshold[neighbour] = 0.0
        return set(currActiveList)

    def getInfectedNodes(self):
        return self.__infectedNodes
