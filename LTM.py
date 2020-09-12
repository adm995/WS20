import networkx as nx
from Similarities import Similarities
import os
from typing import List


class LTM:

    def __init__(self, g: nx.Graph, seeds: List, users: dict, output_dir: str):
        """
        This class provides methods to start a Linear Threshold Influence model on a given graph and a set of seeds to
         start the spread of influence.
        :param g: Users graph
        :param seeds: Set of users in the graph
        :param users: Dictionary that maps each user ID into its visited POIs with relative cts
        :param output_dir: Directory where to write the results
        """
        self.__g = g
        self.__seeds = seeds
        self.__weights = dict()
        self.__threshold = dict()
        self.__actual = dict()
        self.__output = output_dir
        self.__users = users
        self.__active = dict()
        self.__fillThresholds()
        self.__fillWeights()
        self.__computeInfluence()

    def __fillThresholds(self):
        """
        Fill the dictionary that contains the threshold values to be reached by each node in the graph to be active.
        It is 1/degree(node) for each node.
        """
        for node_id in self.__g.nodes:
            self.__actual[node_id] = 0
            self.__threshold[node_id] = 1 / nx.degree(self.__g, node_id)
            self.__active[node_id] = False
            if node_id in self.__seeds:
                self.__active[node_id] = True

    def __fillWeights(self):
        """
        Fill the dictionary that contains the weights between each pair edge of adjacents nodes in the graph.
        The weight for each edge (u, v) is equal to the cosine similarity of the POIs vectors of the two users nodes.
        """
        for user1 in self.__g.nodes:
            for user2 in self.__g.neighbors(user1):

                weight = 0 if (user1 not in self.__users or user2 not in self.__users) else Similarities.get_cosine_similarity(self.__users[user1], self.__users[user2])
                if user1 not in self.__weights:
                    self.__weights[user1] = dict()
                if user2 not in self.__weights:
                    self.__weights[user2] = dict()
                self.__weights[user1][user2] = weight
                self.__weights[user2][user1] = weight

    def __computeInfluence(self):
        """
        Performs the Linear Threshold influence algorithm on the graph.
        """
        print_str = ""
        out_path = os.path.join(self.__output, "influence.txt")
        currActiveNodes = self.__seeds
        for step in range(100):
            print(str(len(currActiveNodes)) + " activated nodes at iteration " + str(step + 1) + " " + str(currActiveNodes) + "\n\n")
            print_str += (str(len(currActiveNodes)) + " new active nodes at iteration " + str(step + 1) + ": " + str(currActiveNodes) + "\n\n")
            for i in range(len(currActiveNodes)):
                activeVertex = currActiveNodes.pop(0)
                for neighbour in self.__g.neighbors(activeVertex):
                    if self.__active[neighbour] is False:
                        self.__actual[neighbour] += self.__weights[activeVertex][neighbour]
                        if self.__actual[neighbour] >= self.__threshold[neighbour]:
                            self.__active[neighbour] = True
                            currActiveNodes.append(neighbour)

        if not os.path.isfile(out_path):
            text_file = open(out_path, "w")
            text_file.write(print_str)
            text_file.close()
        else:
            print("Linear Threshold Influence iterations already written on file")
