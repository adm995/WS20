import networkx as nx

class Connectivity:

    def __init__(self, G):
        self.__G = G

    def connected_component(self, G):
        CCGraphs = list(nx.connected_components(G))
        sortedCC = sorted(CCGraphs, key=nx.density, reverse=True)
        return sortedCC

    def get_topTenBrokers(self, G):
        """
        :param G: Undirected graph
        :return: A dictionary that contains for each CC of G the top 10 brokers in the CC and the relative score.
        """
        d = self.connected_component(G)
        topten = d[:10]
        dicX = {}
        for k in topten:
            dicBcTen = nx.betweenness_centrality(k)
            sortedDicBcTen = sorted(dicBcTen, key=dicBcTen.get, reverse=True)
            dicX[k] = sortedDicBcTen
        return dicX

    def get_topTwentyBrokers(self, G):
        """

        :param G: Undirected graph
        :return:  A dictionary that contains for the CC of G the top 10 brokers in the CC and the relative score.
        """
        d = self.connected_component(G)
        topDense = d[0]
        dicY = {}
        for k in topDense:
            dicBcTwenty = nx.betweenness_centrality(k)
            sortedDicBcTwenty = sorted(dicBcTwenty.values())
            dicY[k] = sortedDicBcTwenty
        return dicY

    def get_topTenDenseCC(self, G):
        f = open('connectivity.txt', 'w')
        d = self.connected_component(G)
        topten = d[:10]
        i = 0
        for k in topten:
            i = i + 1
            f = open('connectivity.txt', 'a')
            f.write(i + " Component: " + nx.nodes(k) + " Density: " + nx.density(k) + "\n")
        f.close()
        return topten








