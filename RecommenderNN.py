import networkx as nx
from sklearn.model_selection import train_test_split
from  keras import *
from typing import Dict
import numpy as np


class RecommenderNN:

    def __init__(self, users: Dict[int, Dict[str, float]], G: nx.Graph):
        self.__friends_avg = 7
        self.__train = self.__preprocess(users, G)

    def __preprocess(self, users, G):
        # al posto di 58000 inserire la somma totale del numero di luoghi visitato da ogni utente
        train_set = np.zeros(shape=(58000, 11))
        return []
