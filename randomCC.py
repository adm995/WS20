from random import randrange, uniform
import networkx as nx
import random


from UsersReader import UsersReader

def main():
    n = random.randint(10, 20)
    print("le componenti connesse saranno: " + str(n))
    finalList = list(range(n))
    lista = list(range(0, 58228))
    print("la lista iniziale ha dimensione: " + str(len(lista)))
    finalList = []
    for i in range(n):
        print("componente connessa numero: " + str(i))
        nodesCClength = random.randint(int((len(lista)/(n-i))/2), int((len(lista)/(n-i))*2))
        if ((nodesCClength>len(lista)) and (i==n-1)):
            nodesCClength = len(lista)
        print("la componente connessa " + str(i) + " , avrà: " + str(nodesCClength) + " nodi " )
        listaCC=[]
        for j in range(nodesCClength):
            if(len(lista)>0):
                nodeCC = random.choice(lista)
                lista.remove(nodeCC)
                listaCC.append(nodeCC)
        finalList.append(listaCC)


    print(n)
    print(len(lista))
    print("------")
    print(len(finalList))
    lineset = []
    f = open("dataset22.txt", "a+")
    for l in finalList:

        print(len(l))
        for id in l:
            a=id
            b=0
            nc = random.randint(1, 17)
            for k in range(nc):
                strA=""
                strB=""
                b = random.choice(l)
                strA = str(a) + "       "  + str(b) + "\n"
                strB = str(b) + "       "  + str(a) + "\n"
                if ((strA not in lineset) and (strB not in lineset)):
                    lineset.append(str(a) + "       "  + str(b) + "\n")
                    f.write(str(a) + "       "  + str(b) + "\n")
    f.close()


if __name__ == '__main__':
    main()
