import networkx as nx
from WS20.Visitors import Visitors


class Main:
    edgesFileName = 'data\loc-brightkite_edges.txt.gz'
    checkinsFilename = 'data\loc-brightkite_totalCheckins.txt.gz'
    """
    Undirected friendship graph: loc - brightkite_edges.txt
    *Nodes: 58228 Edges 214078
    Visitors checkins log: loc - brightkite_totalCheckins.txt
    Locations checkins: 4491143
    """

    def main(self):
        G = nx.read_edgelist(self.edgesFileName, nodetype=int)
        print(G.nodes())

    if __name__ == '__main__':
        v = Visitors(filename='data\loc-brightkite_totalCheckins.txt.gz')
        list = (v.getVisitorsIDs())
        print(list)
