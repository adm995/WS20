import networkx as nx
#from google.colab import drive
from numpy import loadtxt
from UsersReader import UsersReader
from RecommenderNN import RecommenderNN

class Main:
    """
    Undirected friendship graph: loc - brightkite_edges.txt
    Nodes: 58'228 Edges 214'078
    Visitors checkins log: loc - brightkite_totalCheckins.txt
    Locations checkins: 4'491'143
    """


def main():

    #drive.mount('/content/drive')

    edgesFileName_bk = '/content/drive/My Drive/WS2020/loc-brightkite_edges.txt.gz'
    checkinsFilename_bk = '/content/drive/My Drive/WS2020/loc-brightkite_totalCheckins.txt.gz'
    edgesFileName_gw ='/content/drive/My Drive/WS2020/loc-gowalla_edges.txt.gz'
    checkinsFilename_gw ='/content/drive/My Drive/WS2020/loc-gowalla_totalCheckins.txt.gz'
    G = nx.read_edgelist(edgesFileName_gw, nodetype=int)
    users = UsersReader(filename=checkinsFilename_gw)
    mapping = users.getMapper("toID")
    G = nx.relabel_nodes(G, mapping)
    lista = []
    for node in G.nodes:
        lista.append(len(G.edges(node)))
    avg_friends = sum(lista) / (len(lista))
    #print("avg friends:" + str(avg_friends))  # 7 friends
    #print("friends: " + str(sorted(lista, reverse=True)))
    #print("number of users: " + str(len(users.getVisitorsIDs())))
    #print("number of POIs: " + str(users.getPOIcount()))
    #print("Users IDs:" + str(sorted(users.getMap().keys())))
    NN = RecommenderNN(users, G, avg_friends)


if __name__ == '__main__':
    main()
