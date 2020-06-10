import networkx as nx
import math
from numpy import dot
from numpy.linalg import norm

class Similarities:
    def __init__(self, filename, collection):
        self.__filename = filename
        self.__collection = collection  #<int, <string, float> >
        self.__computeSims(self.__collection)

    def get_cosine_similarity(v1, v2):
        "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
        sumxx, sumxy, sumyy = 0, 0, 0
        if len(v1)<=len(v2):
            for i in range(len(v1)):
                x = v1[i]
                y = v2[i]
                sumxx += x * x
                if x in v2:
                   sumyy += y * y
                   sumxy += x * y
            return sumxy / math.sqrt(sumxx * sumyy)
        else:
            for i in range(len(v2)):
                x = v2[i]
                y = v1[i]
                sumxx += x * x
                if x in v1:
                   sumyy += y * y
                   sumxy += x * y
            return sumxy / math.sqrt(sumxx * sumyy)

    def get_jaccard_similarity(v1, v2):
            return  len(v1.keys() & v2.keys()) / float(len(v1.keys() | v2.keys()))

    def __computeSims(self, collection):
        f = open('similarities.txt', 'w')
        for dicLoc1 in collection:
            for dicLoc2 in collection:
                f = open('similarities.txt', 'a')
                f.write("Vis1: " + dicLoc1 + " Vis2: " + dicLoc2 +
                        "Jaccard Sim: " + self.get_jaccard_similarity(collection[dicLoc1], collection[dicLoc2]) +
                        " Cosine Sim: " + self.get_cosine_similarity(collection[dicLoc1], collection[dicLoc2]) + "\n" )
        f.close()



