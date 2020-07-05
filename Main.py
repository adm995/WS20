import networkx as nx
from LTM import LTM
from Visitors import Visitors
import geopy.geocoders
from geopy.geocoders import Nominatim
from functools import partial

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
    print("number of users: "+str(len(v.getVisitorsIDs())))
    print("number of POIs: "+str(v.getPOIcount()))
    ltm = LTM(G, [0, 10, 100], v.getMap(), v.getPOIcount())
    print(ltm.getInfectedNodes())
    geolocator = Nominatim(user_agent="WS2020")
    coords = v.getPOIsCoords()
    reverse = partial(geolocator.reverse, language="en")
    for poi in coords.keys():
        print(reverse(coords[poi]).raw["address"])


if __name__ == '__main__':
    main()
