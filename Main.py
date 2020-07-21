import networkx as nx
from RecommenderNN import RecommenderNN
from numpy import loadtxt
from UsersReader import UsersReader
class Main:
    """
    Undirected friendship graph: loc - brightkite_edges.txt
    Nodes: 58'228 Edges 214'078
    Visitors checkins log: loc - brightkite_totalCheckins.txt
    Locations checkins: 4'491'143
    """


def main():

    edgesFileName = '/content/drive/My Drive/loc-brightkite_edges.txt.gz'
    checkinsFilename = '/content/drive/My Drive/loc-brightkite_totalCheckins.txt.gz'
    G = nx.read_edgelist(edgesFileName, nodetype=int)
    users = UsersReader(filename=checkinsFilename)
    mapping = users.getMapper("toID")
    G = nx.relabel_nodes(G, mapping)
    lista = []
    for node in G.nodes:
        lista.append(len(G.edges(node)))
    print("avg friends:" + str(sum(lista) / (len(lista))))  # 7 friends
    print("friends: " + str(sorted(lista, reverse=True)))
    print("number of users: " + str(len(users.getVisitorsIDs())))
    print("number of POIs: " + str(users.getPOIcount()))
    #print("Users IDs:" + str(sorted(users.getMap().keys())))
    NN = RecommenderNN(users, G)


if __name__ == '__main__':
    main()