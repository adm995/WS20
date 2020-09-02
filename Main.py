import networkx as nx
from numpy import loadtxt
from UsersReader import UsersReader
# from RecommenderNN import RecommenderNN
import json
from Connectivity import Connectivity
import csv
from Preprocessing import Preprocessing

class Main:
    """
    Undirected friendship graph: loc - brightkite_edges.txt
    Nodes: 58'228 Edges 214'078
    Visitors checkins log: loc - brightkite_totalCheckins.txt
    Locations checkins: 4'491'143
    """


def main():

    edgesFileName_gw = '/content/drive/My Drive/WS2020/loc-gowalla_edges.txt.gz'
    checkinsFilename_gw = '/content/drive/My Drive/WS2020/loc-gowalla_totalCheckins.txt.gz'
    watts = "C:/Users/Angelo/Documents/PyCharmProjects/WS20/data/dataset_test.txt.gz"
    G = nx.read_edgelist(watts, nodetype=int)
    print("Graph loaded")
    C = Connectivity(G)
    print(C.connected_component(G))
    users = UsersReader(filename=checkinsFilename_gw)
    mapping = users.getMapper("toID")
    # G = nx.relabel_nodes(G, mapping)  RELABEL NODES FROM 0 TO N
    lista = []
    for node in G.nodes:
        lista.append(len(G.edges(node)))
    avg_friends = sum(lista) / (len(lista))
    print("avg friends:" + str(avg_friends))  # 9 friends
    # print("friends: " + str(sorted(lista, reverse=True)))
    # print("number of users: " + str(len(users.getVisitorsIDs())))
    # print("number of POIs: " + str(users.getPOIcount()))
    # print("Users IDs:" + str(sorted(users.getMap().keys())))
    #NN = RecommenderNN(users, G, avg_friends)

if __name__ == '__main__':
    main()
