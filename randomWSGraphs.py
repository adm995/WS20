import networkx as nx
from UsersReader import UsersReader
from random import randrange, uniform
import networkx as nx
import random
from Connectivity import Connectivity

def main():
    edgesFileName = 'new_dataset.txt'
    G = nx.read_edgelist(edgesFileName, nodetype=int)
    lista = []
    for node in G.nodes:
        lista.append(node)

    ten= int(len(lista)/10)
    i = 0
    j = 0
    print(len(lista))
    print(ten)
    n = 10
    print("le componenti connesse saranno: " + str(n))
    lista = lista
    finalList = []
    for i in range(n):
        nodesCClength = random.randint(int((len(lista) / (n - i)) / 2), int((len(lista) / (n - i)) * 2))
        if ((nodesCClength > len(lista)) and (i == n - 1)):
            nodesCClength = len(lista)
        print("la componente connessa " + str(i) + " , avrà: " + str(nodesCClength) + " nodi ")
        listaCC = []
        for j in range(nodesCClength):
            if (len(lista) > 0):
                nodeCC = random.choice(lista)
                lista.remove(nodeCC)
                listaCC.append(nodeCC)
        finalList.append(listaCC)

    #print(len(finalList))
    f = open('dataset_test2.txt', 'a')
    for l in finalList:
        mapping={}
        j=0
        for a in l:
            mapping[j]=a
            j=j+1
        G = nx.watts_strogatz_graph(len(l), 10, 0.5)
        H = nx.relabel_nodes(G, mapping)
        for coppia in nx.edges(H):
            k=0
            stampa=""
            for nodo1 in coppia:
                if k==0:
                   stampa=stampa+str(nodo1)
                   k=k+1
                else:
                   stampa=stampa+ "  " + str(nodo1) + "\n"
                 #  print(stampa)
                   f.write(stampa)
                   stampa = ""
                   k=0
    f.close()

if __name__ == '__main__':
    main()


