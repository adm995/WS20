import networkx as nx
from sklearn.model_selection import train_test_split
from typing import Dict
import numpy as np
from keras.layers import Embedding, Dense, Dropout, Concatenate, Input, Activation, Lambda, Flatten
from keras.optimizers import Adam
from keras.models import Model
import random
from keras import applications, callbacks, regularizers
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
from UsersReader import UsersReader

class RecommenderNN:

    def __init__(self, users: UsersReader, G: nx.Graph):
        self.__friends_avg = 7
        self.__POI_count = users.getPOIcount()
        self.__n_users = len(users.getVisitorsIDs())
        # self.__all_data = self.__preprocess(users, G, self.__friends_avg)
        self.__all_data = loadtxt('/content/drive/My Drive/WS2020_NN_input.csv', delimiter=',')
        self.__train_model()

    def __preprocess(self, users: UsersReader, g: nx.Graph, friends_number: int):
        """

        :param users: Users read from dataset
        :param g: friendship graph
        :return: input matrix for the neural network
        """
        # print("START")
        total_visits = []
        ids = users.getVisitorsIDs()
        for user_id in ids:
            # print(user_id)
            total_visits.append(len(users.getVisitorPOIs(user_id)))
        # print("OK")
        rows_number = sum(total_visits)
        # without poi_category
        cols_number = 3 + friends_number  # user_id, poi_id, friends_cts, user_cts
        all_data = np.zeros((rows_number, cols_number))
        row = 0
        for user_id in users.getVisitorsIDs():
            # print(user_id)
            for poi_id in users.getVisitorPOIs(user_id):
                all_data[row, 0] = user_id
                all_data[row, 1] = poi_id
                # all_data[row, 2] = 0 #category
                user_friends = set(g[user_id].keys())
                train_friends = user_friends.intersection(users.getPOIvisitors(poi_id))

                for col in range(3, 10):
                    if len(train_friends) > 0:
                        train_friends = list(train_friends)
                        all_data[row, col] = train_friends.pop(random.randint(0, len(train_friends) - 1))
                    else:
                        all_data[row, col] = 0
                all_data[row, 9] = users.getVisitorCts(user_id, poi_id)
                row += 1

        data = asarray(all_data)
        # save to csv file
        savetxt('/content/drive/My Drive/WS2020_NN_input.csv', data, delimiter=',')

        return all_data

    def __RecommenderNet(self, n_users, n_pois, n_factors, min_rating, max_rating, user_shape, poi_shape,
                         friends_shape) -> Model:
        """
        Return the neural network model ready to be trained on processed input data.
        :param n_users: number of users
        :param n_pois: number of POIs
        :param n_factors:
        :param min_rating: min cts
        :param max_rating: max cts
        :return: NN model
        """
        user = Input(shape=user_shape, name="User_ID")
        user_embedding = Embedding(n_users + 1, n_factors)(user)
        user_vec = Flatten(name='FlattenUsers')(user_embedding)

        poi = Input(shape=poi_shape, name="poi_ID")
        poi_embedding = Embedding(n_pois + 1, n_factors)(poi)
        poi_vec = Flatten(name='FlattenPois')(poi_embedding)

        friends = Input(shape=friends_shape, name="friends")
        # friends_vec = Flatten(name='FlattenFriends')(friends)
        # f = Embedding(7, 7)(friends)

        x = Concatenate(axis=1)([user_vec, poi_vec, friends])
        x = Dropout(0.2)(x)

        x = Dense(20, kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)

        x = Dense(1, kernel_initializer='he_normal')(x)
        x = Activation('sigmoid')(x)
        x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)
        model = Model(inputs=[user, poi, friends], outputs=x)
        opt = Adam(lr=0.1)
        model.compile(loss='mean_squared_error', optimizer=opt)
        model.summary()
        return model

    def __train_model(self):
        n_pois = self.__POI_count
        batch_size = 32

        X = np.array(self.__all_data[:, : 9])
        y = np.array(self.__all_data[:, 9])
        y = np.interp(y, (y.min(), y.max()), (0, 5))
        print("Intepolated y: " + str(y[0]))
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, test_size=0.4, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=42)
        stopping = callbacks.EarlyStopping(monitor='val_acc', patience=25, restore_best_weights=True)

        X_train_users = np.array(X_train[:, 0])
        X_train_pois = np.array(X_train[:, 1])
        X_train_friends = np.array(X_train[:, 2:])

        X_train_users = np.reshape(X_train_users, (645700, 1))
        X_train_pois = np.reshape(X_train_pois, (645700, 1))
        X_train_friends = np.reshape(X_train_friends, (645700, 7))
        y_train = np.reshape(y_train, (645700, 1))

        # X_train_users = np.expand_dims(X_train_users, axis=0)
        # X_train_pois = np.expand_dims(X_train_pois, axis=0)
        # X_train_friends = np.expand_dims(X_train_friends, axis=0)

        X_val_users = np.array(X_val[:, 0])
        X_val_pois = np.array(X_val[:, 1])
        X_val_friends = np.array(X_val[:, 2:])

        X_val_users = np.reshape(X_val_users, (322850, 1))
        X_val_pois = np.reshape(X_val_pois, (322850, 1))
        X_val_friends = np.reshape(X_val_friends, (322850, 7))
        y_val = np.reshape(y_val, (322850, 1))

        # X_val_users = np.expand_dims(X_val_users, axis=0)
        # X_val_pois = np.expand_dims(X_val_pois, axis=0)
        # X_val_friends = np.expand_dims(X_val_friends, axis=0)

        print("user_poi_friends_train shape: " + "(" + str(X_train_users[0].shape) + ", " + str(
            X_train_pois[0].shape) + ", " + str(X_train_friends[0].shape) + ")")
        print("user_poi_friends_train shape: " + "(" + str(X_val_users[0].shape) + ", " + str(
            X_val_pois[0].shape) + ", " + str(X_val_friends[0].shape) + ")")
        print("y_train, y_val shape:" + str(y_train.shape) + ", " + str(y_val.shape))
        model = self.__RecommenderNet(self.__n_users, n_pois, 7, 1, 5, X_train_users[0].shape, X_train_pois[0].shape,
                                      X_train_friends[0].shape)
        model.fit([X_train_users, X_train_pois, X_train_friends], y_train, batch_size=64, epochs=100,
                  callbacks=[stopping],
                  validation_data=([X_val_users, X_val_pois, X_val_friends], y_val), verbose=1)
