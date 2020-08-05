import networkx as nx
import operator
from collections import OrderedDict
import itertools
import collections


class Connectivity:

    def __init__(self, G):
        self.__G = G

    def connected_component(self, G):
        CCGraphs = list(nx.connected_components(G))
        CCGraphs.sort(key=len, reverse=True)
        return CCGraphs
        # sortedCC = sorted(CCGraphs, key=nx.density, reverse=True)



    def get_topTenBrokers(self, G):
        """
        :param G: Undirected graph
        :return: A dictionary that contains for each CC of G the top 10 brokers in the CC and the relative score.
        Find the top 10 brokers in the top 10 most dense connected components."""

        toptenCC = self.get_topTenDenseCC(G)

        toptenBro = {}
        for k in toptenCC:
            S = G.subgraph(k).copy()
            dicBcTen = nx.betweenness_centrality(S)
            sortedDicBcTen = OrderedDict(sorted(dicBcTen.items(), key=lambda x: x[1], reverse=True))
            ii = 0
            dicX = {}
            #print("len prima: " + str(len(dicX)))
            for broker in sortedDicBcTen:
                ii = ii + 1
                if ii <= 10:
                    dicX[broker] = sortedDicBcTen[broker]
                    #print("bro " + str(broker) + " numero " + str(ii))
                else:
                    toptenBro[tuple(k)] = dicX
                    #print("len dopo: " + str(len(dicX)))
                    break
        return toptenBro

    def get_topTenDenseCC(self, G):
        d = self.connected_component(G)
        dizionario = {}
        topten = {}
        i = 0
        for k in d:
            S = G.subgraph(k).copy()
            i = i + 1
            dizionario[tuple(k)] = nx.density(S)
            print(" Component number:  " + str(i) + " Density: " + str(nx.density(S)) + " contain: " + str(
                nx.nodes(S)) + "\n")
        # print(" grandezza dizionario:  " + str(len(dizionario)) + "\n")
        sortedDict = OrderedDict(sorted(dizionario.items(), key=lambda x: x[1], reverse=True))
        f = open('connectivity.txt', 'w')
        i = 0
        for k in sortedDict:
            i = i + 1
            if i <= 10:
                # print(" Component number:  " + str(i)  + " key: " + str(k) )
                # print(" value: " + str(sortedDict[k]) + "\n")
                topten[tuple(k)] = sortedDict.get(k)
                f.write(" Component number:  " + str(i) + " Density: " + str(sortedDict.get(k)) + " contain: \n")
            else:
                break

        f.close()
        return topten

    def get_topOneDenseCC(self, G):
        d = self.connected_component(G)
        dizionario = {}
        topOne = {}
        i = 0
        for k in d:
            S = G.subgraph(k).copy()
            i = i + 1
            dizionario[tuple(k)] = nx.density(S)
            print(" Component number:  " + str(i) + " Density: " + str(nx.density(S)) + " contain: " + str(
                nx.nodes(S)) + "\n")
        # print(" grandezza dizionario:  " + str(len(dizionario)) + "\n")
        sortedDict = OrderedDict(sorted(dizionario.items(), key=lambda x: x[1], reverse=True))
        i = 0
        for k in sortedDict:
            i = i + 1
            if i == 1:
                # print(" Component number:  " + str(i)  + " key: " + str(k) )
                # print(" value: " + str(sortedDict[k]) + "\n")
                topOne[tuple(k)] = sortedDict.get(k)
                return topOne
            else:
                break
        return topOne

    def get_topTwentyBrokers(self, G):
        """:param G: Undirected graph
         :return: A dictionary that contains for each CC of G the top 10 brokers in the CC and the relative score.
        Find the top 10 brokers in the top 10 most dense connected components."""
        topOneCC = self.get_topOneDenseCC(G)
        toptwentyBro = {}
        for k in topOneCC:
            S = G.subgraph(k).copy()
            dicBcTwe = nx.betweenness_centrality(S)
            sortedDicBcTwe = OrderedDict(sorted(dicBcTwe.items(), key=lambda x: x[1], reverse=True))
            ii = 0
            dicX = {}
            #print("len prima: " + str(len(dicX)))
            for broker in sortedDicBcTwe:
                ii = ii + 1
                if ii <= 20:
                    dicX[broker] = sortedDicBcTwe[broker]
                    #print("bro " + str(broker) + " numero " + str(ii))
                else:
                    toptwentyBro[tuple(k)] = dicX
                    #print("len dopo: " + str(len(dicX)))
                    break
        return toptwentyBro
