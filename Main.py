import networkx as nx
from UsersReader import UsersReader
from RecommenderNN import RecommenderNN
from Connectivity import Connectivity
from Similarities import Similarities
from LTM import LTM
import os
import gzip


class Main:

    def __init__(self):
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    def start(self):

        output_dir = os.path.join(self.ROOT_DIR, "output")
        data_dir = os.path.join(self.ROOT_DIR, "data")

        edgesFileName_gw_small_random = os.path.join(data_dir, "gowalla_small_graph_random.txt.gz")
        edgesFileName_gw_small_original = os.path.join(data_dir, "loc-gowalla_edges.txt.gz")
        edgesFileName_gw_full_random = os.path.join(data_dir, "gowalla_full_graph_random.txt.gz")
        edgesFileName_gw_full = os.path.join(data_dir, "gowalla_full_graph.csv.gz")

        checkinsFilename_gw_small_cat = os.path.join(data_dir, "checkins_small_cat.txt.gz")
        checkinsFilename_gw_full_cat = os.path.join(data_dir, "checkins_full_cat.txt.gz")

        checkins = checkinsFilename_gw_small_cat
        edges = edgesFileName_gw_small_random

        if ".csv" in edges:
            with gzip.open(edges, 'rb') as edgescsv:
                next(edgescsv, '')  # skip headers line
                G = nx.read_edgelist(edgescsv, nodetype=int, delimiter=",")
            edgescsv.close()
        else:
            G = nx.read_edgelist(edges, nodetype=int)

        is_full = "full" in checkins

        # TASK 1 user representation: UsersReader creates a dictionary that maps each user ID in its list of POIs and
        # the relative CTS: DONE
        users = UsersReader(filename=checkins, G=G)

        # TASK 2a similarities: Print Cosine/Jaccard similarity for each pair of nodes: DONE
        Similarities(output_dir, users.getMap(), G)

        # TASK 2b: select top 10 dense CC: DONE
        c = Connectivity(G, output_dir)
        topDenseCC = c.getMostDenseCC()[0]

        # TASK 2c: find top 10 brokers in top 10 dense CC: DONE
        c.getTopTenBrokers()

        # TASK 2d: from top 20 brokers in the most dense CC apply LTM: DONE
        top20brokers = c.getTopTwentyBrokers()
        LTM(topDenseCC, top20brokers[1], users.getMap(), output_dir)

        # TASK 3 and 4: create and evaluate the recommender system: DONE
        nodes = []
        for node in G.nodes:
            nodes.append(len(G.edges(node)))
        avg_friends = 1 if is_full else round((sum(nodes) / (len(nodes))) / 2)
        RecommenderNN(users, G, avg_friends, is_full, data_dir)


if __name__ == '__main__':
    Main = Main()
    Main.start()
