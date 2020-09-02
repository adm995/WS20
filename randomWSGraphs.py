from UsersReader import UsersReader
from random import randrange, uniform
import networkx as nx
import random
import csv
from Connectivity import Connectivity


def main():
    edgesFileName = '/Users/Angelo/Desktop/gowalla/gowalla_friendship.csv'
    G = None
    with open(edgesFileName, 'rb') as edges:
        next(edges, '')   # skip a line
        G = nx.read_edgelist(edges, nodetype=int, delimiter=",")
    edges.close()
    node_list = []
    for node in G.nodes:
        node_list.append(node)

    components = int(len(node_list)/10)
    print(len(node_list))
    print(components)
    CC_number = 30
    print("There will be "+str(CC_number)+" connected components")
    final_CC_list = []
    for i in range(CC_number):
        nodesCClength = random.randint(int((len(node_list) / (CC_number - i)) / 2), int((len(node_list) / (CC_number - i)) * 2))
        if nodesCClength > len(node_list) and (i == CC_number - 1):
            nodesCClength = len(node_list)
        print("The CC " + str(i) + " , will have: " + str(nodesCClength) + " nodes")
        CC = random.sample(node_list, nodesCClength)
        final_CC_list.append(CC)

    f = open('dataset_test2.txt', 'a')
    for CC in final_CC_list:
        mapping = {}
        new_id = 0
        for node in CC:
            mapping[new_id] = node
            new_id = new_id+1
        G = nx.watts_strogatz_graph(len(CC), 30, 0.5)
        H = nx.relabel_nodes(G, mapping)
        for pair in nx.edges(H):
            k = 0
            print_str = ""
            for node1 in pair:
                if k == 0:
                    print_str = print_str+str(node1)
                    k = k+1
                else:
                    print_str = print_str + "  " + str(node1) + "\n"
                    f.write(print_str)
                    print_str = ""
                    k = 0
    f.close()


if __name__ == '__main__':
    main()
