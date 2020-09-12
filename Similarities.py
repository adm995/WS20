import math
import os
from typing import Dict
import csv
import networkx as nx


class Similarities:

    def __init__(self, output_dir: str, users: Dict, G: nx.Graph):
        """
        This class provides methods to measure Jaccard and Cosine similarities among all pair of users inside
        the dataset.
        :param output_dir: Directory where to write the output files
        :param users: Dictionary that maps each user ID into its visited POIs with relative cts
        :param G: Friendship graph
        """
        self.__G = G
        self.__output_dir = output_dir
        self.__users = users  # <user_id: int, <poi_id: str, cts: float> >
        self.__computeSims(self.__users)

    @staticmethod
    def get_cosine_similarity(v1: Dict, v2: Dict):
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
    def get_jaccard_similarity(v1: Dict, v2: Dict):
        """
        Compute Jaccard similarity of v1 to v2.
        :param v1: Dictionary that maps for each POI the relative cts for visitor v1.
        :param v2: Dictionary that maps for each POI the relative cts for visitor v2.
        :return: The Jaccard similarity between v1 and v2.
        """
        return len(v1.keys() & v2.keys()) / float(len(v1.keys() | v2.keys()))

    def __computeSims(self, users: Dict):
        """
        Print on a csv file the Jaccard and cosine similarities between each user and its friends.
        :param users: the dictionary that contains all users in the dataset and their visited POIs
        :return:
        """
        out_path = os.path.join(self.__output_dir, "similarities.csv")
        if not os.path.isfile(out_path):
            with open(out_path, 'w',  newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["V1", "V2", "Jac", "Cos"])
                users_id = sorted(users.keys())
                row = 1
                i = 0
                for user1_id in users_id:
                    i += 1
                    print("User "+str(i)+" of "+str(len(users_id)))
                    for user2_id in self.__G[user1_id]:
                        if user2_id in users:
                            jac = self.get_jaccard_similarity(users[user1_id], users[user2_id])
                            cos = self.get_cosine_similarity(users[user1_id], users[user2_id])
                            writer.writerow([row, str(user1_id), str(user2_id), str(jac), str(cos)])
                            row += 1
        else:
            print("Similarities already written on file")
