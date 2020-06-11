import networkx as nx

class Connectivity:

    def __init__(self, G):
        self.__G=G

    def connected_component(G):
        dic = {}
        CCGraphs = list(nx.connected_component_subgraphs(G))
        for cc in CCGraphs:
         dic[cc]=nx.density(cc)
        sortedDict = sorted(dic.values())
        return sortedDict

    def get_topTenBrokers(self, G, d):
        d = self.connected_component(G)
        topten = dict(d.items()[:10])
        dicX = {}
        for k,v in topten:
            dicBcTen = nx.betweenness_centrality(k)
            sortedDicBcTen = sorted(dicBcTen.values())
            dicX[k]=sortedDicBcTen
        return dicX

    def get_topTwentyBrokers(self, G, d):
        d = self.connected_component(G)
        toptwenty = dict(d.items()[:20])
        dicY = {}
        for k, v in toptwenty:
            dicBcTwenty = nx.betweenness_centrality(k)
            sortedDicBcTwenty = sorted(dicBcTwenty.values())
            dicY[k] = sortedDicBcTwenty
        return dicY

    def get_topTenDenseCC(self, G, d):
        f = open('connectivity.txt', 'w')
        d = self.connected_component(G)
        topten = dict(d.items()[:10])
        i = 0
        for k, v in topten:
            i = i + 1
            f = open('connectivity.txt', 'a')
            f.write(i + " Component: " + nx.nodes(k) + " Density: " + v + "\n")
        f.close()
        return topten








