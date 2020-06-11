import networkx as nx

from WS20.LTM import LTM
from WS20.Visitors import Visitors
import numpy as np
from scipy.spatial import distance


class Main:

    """
    Undirected friendship graph: loc - brightkite_edges.txt
    Nodes: 58228 Edges 214078
    Visitors checkins log: loc - brightkite_totalCheckins.txt
    Locations checkins: 4491143
    """

    def main():
        edgesFileName = 'data\loc-brightkite_edges.txt.gz'
        checkinsFilename = 'data\loc-brightkite_totalCheckins.txt.gz'
        G = nx.read_edgelist(edgesFileName, nodetype=int)
        v = Visitors(filename=checkinsFilename)
        mapping = v.getMapper("toID")
        G = nx.relabel_nodes(G, mapping)
        m = v.getMap()
        print(sorted(m.keys()))
        print(sorted(G))
        ltm = LTM(G, [0, 10, 100], v.getMap(), v.getPOIcount())
        print(ltm.getInfectedNodes())

    if __name__ == '__main__':
        main()


