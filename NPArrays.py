import numpy as np

class NPArrays:

    def __init__(self, filename, collection, g: nx.Graph):
        self.__filename = filename
        self.__collection = collection
        self.__g = g
        self.__computeMatrix(self.__collection, self.__g)



        def __computeMatrix(self, collection, g):
            mylist = []
            for dicUtente in collection:
                for dicLuogo in dicUtente:
                    mylist.insert(dicLuogo)
            n= len(mylist)
            m=11
            M = np.zeros((n,m))
            for dicUtente in collection:
                a=0
                for dicLuogo in dicUtente:
                    M[a,0] = dicUtente
                    M[a,1] = dicLuogo
                    M[a,2] = "categoria"
                    for node in g.nodes:
                        if(node==dicUtente):
                            lista = []
                            lista.append((g.edges(node)))
                            for friend in lista:
                                if not (friend in dicUtente and dicLuogo == dicUtente[friend]):
                                    lista.remove(friend)
                            for n in range(3,10):
                                if(len(lista)>0):
                                    M[a,n] = lista.pop(random.randint(0,len(lista)-1))[dicLuogo]
                                else:
                                    M[a, n] = "/"
                    M[a,10] = dicUtente[dicLuogo]
                    a=a+1
            return M
