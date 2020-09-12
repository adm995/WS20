import networkx as nx
import os
import csv
from typing import List


class Connectivity:

    def __init__(self, G: nx.Graph, output_dir: str):
        """
        This class provides methods get the most dense connected components in the input graph and to select the main
        brokers inside the CC.
        :param G: Undirected graph
        :param output_dir: Output directory
        """
        self.__output_dir = output_dir
        self.__G = G
        self.__CC = None

    def getMostDenseCC(self) -> List[nx.Graph]:
        """
        Find the top 10 most denseCC in the graph.
        :return: The top 10 most denseCC in the graph.
        """
        S = [self.__G.subgraph(c).copy() for c in nx.connected_components(self.__G)]
        sortedCC = sorted(S, key=nx.density, reverse=True)
        self.__CC = sortedCC
        return sortedCC

    def getTopTenBrokers(self):
        """
        Find the top 10 dense CC of the graph and their top 10 brokers with the relative centrality score.
        :return: A tuple that contains for the top 10 dense CC of G their top 10 brokers and the relative score.
        """
        d = self.getMostDenseCC() if self.__CC is None else self.__CC
        topten = d[:10]
        CC_dic = {}
        cc_num = 1
        print_str = ""
        for CC in topten:
            print_str += "Top 10 brokers for component "+str(cc_num)+"\n"
            nodes_centrality_dict = nx.degree_centrality(CC)
            nodes_id = sorted(nodes_centrality_dict, key=nodes_centrality_dict.get, reverse=True)
            brokers_scores = []
            for rank in range(10):
                broker_id = nodes_id[rank]
                score = nodes_centrality_dict[broker_id]
                brokers_scores.append(score)
                print_str += "At rank "+str(rank)+" broker with ID "+str(broker_id)+" has score:"+str(score)+"\n"

            CC_dic[cc_num] = (brokers_scores, nodes_id[:10])
            cc_num += 1

        out_path = os.path.join(self.__output_dir, "top10inTop10.txt")
        if not os.path.isfile(out_path):
            text_file = open(out_path, "w")
            text_file.write(print_str)
            text_file.close()
        else:
            print("Top 10 brokers of top 10 most dense CC already written on file")
        return CC_dic

    def getTopTwentyBrokers(self):
        """
        :return: A tuple that contains for the most dense CC of G the top 20 brokers in the CC and the relative
        scores.
        """
        d = self.getMostDenseCC() if self.__CC is None else self.__CC
        topDense = d[0]
        nodes_centrality_dict = nx.degree_centrality(topDense)
        nodes_id = sorted(nodes_centrality_dict, key=nodes_centrality_dict.get, reverse=True)
        print_str = "Top 20 brokers in the most dense CC: \n"
        brokers_scores = []
        for rank in range(20):
            broker_id = nodes_id[rank]
            score = nodes_centrality_dict[broker_id]
            brokers_scores.append(score)
            print_str += "At rank " + str(rank) + " broker with ID " + str(broker_id) + " has score:" + str(score)+"\n"

        out_path = os.path.join(self.__output_dir, "top20.txt")
        if not os.path.isfile(out_path):
            text_file = open(out_path, "w")
            text_file.write(print_str)
            text_file.close()
        else:
            print("Top 20 brokers of the most dense CC already written on file")

        return brokers_scores, nodes_id[:20]

    def printTopTenDenseCC(self):
        """
        Print on a csv file the connected components in descending order of density.
        """
        out_path = os.path.join(self.__output_dir, "CC_density.csv")
        d = self.getMostDenseCC() if self.__CC is None else self.__CC
        topten = d[:10]
        if not os.path.isfile(out_path):
            print("Print CC nodes, density on file")
            with open(out_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Component", "Density"])
                row = 1
                for k in topten:
                    writer.writerow([row, str(nx.nodes(k)), nx.density(k)])
                    row += 1
            csvfile.close()
        else:
            print("Top 10 most dense CC already written on file")
