import networkx as nx
from sklearn.model_selection import train_test_split
from typing import Dict
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, Concatenate, Lambda
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam


class RecommenderNN:

    def __init__(self, users: Dict[int, Dict[str, float]], G: nx.Graph):
        self.__friends_avg = 7
        self.__all_data = self.__preprocess(users, G)

    def __preprocess(self, users, G):
        # al posto di 58000 inserire la somma totale del numero di luoghi visitato da ogni utente
        train_set = np.zeros(shape=(58000, 11))
        return []

    def __RecommenderNet(self, n_users, n_pois, n_factors, min_rating, max_rating):
        """
        Return the neural network model ready to be trained on processed input data.
        :param n_users: number of users
        :param n_pois: number of POIs
        :param n_factors:
        :param min_rating: min cts
        :param max_rating: max cts
        :return: NN model
        """
        user = Input(shape=(1,))
        u = Embedding(n_users, n_factors)(user)

        poi = Input(shape=(1,))
        p = Embedding(n_pois, n_factors)(poi)

        friends = Input(shape=(1,))
        f = Embedding(n_pois, n_factors)(friends)

        x = Concatenate()([u, p, f])
        x = Dropout(0.05)(x)

        x = Dense(10, kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(1, kernel_initializer='he_normal')(x)
        x = Activation('sigmoid')(x)
        x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)
        model = Model(inputs=[user, poi, friends], outputs=x)
        opt = Adam(lr=0.001)
        model.compile(loss='mean_squared_error', optimizer=opt)
        return model

    def __train(self):
        X = self.__all_data[:, 10]
        y = self.__all_data[:, 10]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, test_size=0.4, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=42)