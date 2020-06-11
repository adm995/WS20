import math


class Similarities:

    def __init__(self, filename, collection):
        self.__filename = filename
        self.__collection = collection  # <int, <string, float> >
        self.__computeSims(self.__collection)

    @staticmethod
    def get_cosine_similarity(v1, v2):
        """
        Compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
        :param v1: Dictionary that maps for each POI the relative cts for visitor v1.
        :param v2: Dictionary that maps for each POI the relative cts for visitor v2.
        :return: The cosine similarity between v1 and v2.
        """
        sumxx, sumxy, sumyy = 0, 0, 0
        if len(v1) <= len(v2):
            for i in v1.keys():
                if i in v2.keys():
                    x = v1[i]
                    y = v2[i]
                    sumxx += x * x
                    sumyy += y * y
                    sumxy += x * y
            den = math.sqrt(sumxx * sumyy)
            if den != 0:
                return sumxy / (math.sqrt(sumxx) * math.sqrt(sumyy))
            else:
                return 0
        else:
            for i in v2.keys():
                if i in v1.keys():
                    x = v2[i]
                    y = v1[i]
                    sumxx += x * x
                    sumyy += y * y
                    sumxy += x * y
            den = math.sqrt(sumxx * sumyy)
            if den != 0:
                return sumxy / (math.sqrt(sumxx) * math.sqrt(sumyy))
            else:
                return 0

    @staticmethod
    def get_jaccard_similarity(v1, v2):
        """
        Compute Jaccard similarity of v1 to v2.
        :param v1: Dictionary that maps for each POI the relative cts for visitor v1.
        :param v2: Dictionary that maps for each POI the relative cts for visitor v2.
        :return: The Jaccard similarity between v1 and v2.
        """
        return len(v1.keys() & v2.keys()) / float(len(v1.keys() | v2.keys()))

    def __computeSims(self, collection):
        f = open('similarities.txt', 'w')
        for dicLoc1 in collection:
            for dicLoc2 in collection:
                f = open('similarities.txt', 'a')
                f.write("Vis1: " + dicLoc1 + " Vis2: " + dicLoc2 +
                        "Jaccard Sim: " + str(self.get_jaccard_similarity(collection[dicLoc1], collection[dicLoc2])) +
                        " Cosine Sim: " + str(self.get_cosine_similarity(collection[dicLoc1], collection[dicLoc2])) +
                        "\n")
        f.close()
