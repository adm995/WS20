import networkx as nx
import random
import gzip
import os


def createRandomGraph():
    """
    Create a random graph made up of 10 CC obtained partitioning the set of nodes of another existing graph and then
    adding (with 0.5 probability for each one) 30 edges from each node toward other nodes inside the same CC.
    :return:
    """
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(ROOT_DIR, "output")
    data_dir = os.path.join(ROOT_DIR, "data")

    edgesFileName = os.path.join(data_dir, "loc-gowalla_edges.txt.gz")
    if ".csv" in edgesFileName:
        with gzip.open(edgesFileName, 'rb') as edgescsv:
            next(edgescsv, '')  # skip headers line
            G = nx.read_edgelist(edgescsv, nodetype=int, delimiter=",")
        edgescsv.close()
    else:
        G = nx.read_edgelist(edgesFileName, nodetype=int)

    node_list = set()
    for node in G.nodes:
        node_list.add(node)

    components = int(len(node_list)/10)
    print(components)
    CC_number = 10
    final_CC_list = []
    for i in range(CC_number):
        nodesCClength = random.randint(int((len(node_list) / (CC_number - i)) / 2), int((len(node_list) / (CC_number - i)) * 2))
        if (nodesCClength > len(node_list) and (i == CC_number - 1)) or i == 9:
            nodesCClength = len(node_list)
        CC = random.sample(node_list, nodesCClength)
        node_list = node_list - set(CC)
        final_CC_list.append(CC)

    out_path = os.path.join(output_dir, "gowalla_small_graph_random.txt")
    if not os.path.isfile(out_path):
        f = open(out_path, 'a')
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
    else:
        print("The random graph is already present")


if __name__ == '__main__':
    createRandomGraph()
