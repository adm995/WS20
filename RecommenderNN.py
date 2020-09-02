import networkx as nx
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Embedding, Dense, Dropout, Concatenate, Input, Activation, Flatten
from keras.optimizers import Adam, Adadelta
from keras.models import Model
from keras.regularizers import l2
import random
from keras import callbacks
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
from keras.models import load_model
from UsersReader import UsersReader


class RecommenderNN:

    def __init__(self, users: UsersReader, G: nx.Graph, friends_avg: float):
        self.__users = users
        self.__friends_avg = round(friends_avg)
        self.__POI_count = users.getPOIcount()
        self.__POI_ids = users.getPOIsIDs()
        self.__category_to_id = dict()
        ind = 0
        self.__POI_category = users.getCategories()
        for cat in self.__POI_category.values():
            self.__category_to_id[cat] = ind
            ind += 1
        self.__mapper = users.getMapper("toID")
        if os.path.isfile('/content/drive/My Drive/WS2020/WS2020_gw_category_NN_input_mean.csv'):
            self.__all_data = loadtxt('/content/drive/My Drive/WS2020/WS2020_gw_category_NN_input_mean.csv',
                                      delimiter=',')
        else:
            self.__all_data = self.__preprocess(users, G, self.__friends_avg)
        self.__n_users = int(max(self.__all_data[:, 0]))
        self.__indexes_dict = dict()
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
        cols_number = 4 + friends_number  # user_id, poi_id, poi_category, friends_cts, user_cts
        all_data = np.zeros((rows_number, cols_number))
        row = 0
        print("n_users: " + str(max(ids)))
        for user_id in ids:
            print(user_id)
            friends_cts = []
            for poi_id in users.getVisitorPOIs(user_id):
                all_data[row, 0] = self.__mapper[str(user_id)]
                all_data[row, 1] = poi_id
                category_name = self.__POI_category[poi_id]
                all_data[row, 2] = self.__category_to_id[category_name]
                user_friends = set(g[user_id].keys())

                # print("Friends of user "+str(user_id)+", poi: "+str(poi_id))
                common_poi_friends = user_friends.intersection(users.getPOIvisitors(poi_id))

                for friend_id in common_poi_friends:
                    friends_cts.append(users.getVisitorCts(friend_id, poi_id))
                avg_friends_rating = 0 if len(friends_cts) == 0 else (sum(friends_cts) / len(friends_cts))
                all_data[row, 3] = avg_friends_rating
                all_data[row, -1] = users.getVisitorCts(user_id, poi_id)
                row += 1

        data = asarray(all_data)
        # save to csv file
        savetxt('/content/drive/My Drive/WS2020/WS2020_gw_category_NN_input_mean.csv', data, delimiter=',')

        return all_data

    def __load_data(self, friends_avg: int):
        sc = MinMaxScaler()
        X = np.array(self.__all_data[:, : 3 + friends_avg])
        # X = sc.fit_transform(X)
        y = np.array(self.__all_data[:, 3 + friends_avg])
        y = y.reshape(-1, 1)
        y = sc.fit_transform(y)
        print(y[0])
        data = []
        # y = np.interp(y, (y.min(), y.max()), (0, 5))
        # print("Intepolated y: "+str(y[0]))
        print("min rating: " + str(min(y)))
        print("max rating: " + str(max(y)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, test_size=0.4, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

        X_train_users = np.array(X_train[:, 0])
        X_train_pois = np.array(X_train[:, 1])
        X_train_category = np.array(X_train[:, 2])
        X_train_friends = np.array(X_train[:, 3])
        train_samples = X_train_users.shape[0]

        X_train_users = np.reshape(X_train_users, (train_samples, 1))
        X_train_pois = np.reshape(X_train_pois, (train_samples, 1))
        X_train_category = np.reshape(X_train_category, (train_samples, 1))
        # ValueError: cannot reshape array of size 66873037 into shape (6079367,1)
        X_train_friends = np.reshape(X_train_friends, (train_samples, 1))
        y_train = np.reshape(y_train, (train_samples, 1))
        data.append((X_train_users, X_train_pois, X_train_category, X_train_friends, y_train))

        X_val_users = np.array(X_val[:, 0])
        X_val_pois = np.array(X_val[:, 1])
        X_val_category = np.array(X_val[:, 2])
        X_val_friends = np.array(X_val[:, 3])
        val_samples = X_val_users.shape[0]

        X_val_users = np.reshape(X_val_users, (val_samples, 1))
        X_val_pois = np.reshape(X_val_pois, (val_samples, 1))
        X_val_category = np.reshape(X_val_category, (val_samples, 1))
        X_val_friends = np.reshape(X_val_friends, (val_samples, 1))
        y_val = np.reshape(y_val, (val_samples, 1))
        data.append((X_val_users, X_val_pois, X_val_category, X_val_friends, y_val))

        X_test_users = np.array(X_test[:, 0])
        X_test_pois = np.array(X_test[:, 1])
        X_test_category = np.array(X_test[:, 2])
        X_test_friends = np.array(X_test[:, 3])
        test_samples = X_test_users.shape[0]

        X_test_users = np.reshape(X_test_users, (test_samples, 1))
        X_test_pois = np.reshape(X_test_pois, (test_samples, 1))
        X_test_category = np.reshape(X_test_category, (test_samples, 1))
        X_test_friends = np.reshape(X_test_friends, (test_samples, 1))
        y_test = np.reshape(y_test, (test_samples, 1))
        data.append((X_test_users, X_test_pois, X_test_category, X_test_friends, y_test))

        return data

    def __RecommenderNet(self, n_users, n_pois, n_factors, user_shape, poi_shape,
                         category_shape, friends_shape) -> Model:
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
        user_embedding = Embedding(n_users + 1, n_factors, embeddings_initializer='he_normal')(user)
        user_vec = Flatten(name='FlattenUsers')(user_embedding)

        poi = Input(shape=poi_shape, name="poi_ID")
        poi_embedding = Embedding(n_pois + 1, n_factors, embeddings_initializer='he_normal')(poi)
        poi_vec = Flatten(name='FlattenPois')(poi_embedding)

        category = Input(shape=category_shape, name="poi_category")
        # category_embedding = Embedding(n_categories + 1, n_factors, embeddings_initializer='he_normal')(category)
        category_vec = Flatten(name='FlattenCategories')(category)

        friends = Input(shape=friends_shape, name="friends")
        friends_vec = Flatten(name='FlattenFriends')(friends)
        # f = Embedding(7, 7)(friends)
        # dot_user_poi = Dot(axes=1, normalize=False, name = "Dot_user_poi")([user, poi])
        x = Concatenate(axis=1)([user_vec, poi_vec, category_vec, friends_vec])
        x = Dropout(0.5)(x)

        x = Dense(10, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.001))(x)
        x = Activation('relu')(x)
        x = Dropout(0.7)(x)

        x = Dense(5, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.001))(x)
        x = Activation('relu')(x)
        x = Dropout(0.7)(x)

        x = Dense(1, kernel_initializer='he_uniform')(x)
        x = Activation('sigmoid')(x)
        # x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)
        model = Model(inputs=[user, poi, category, friends], outputs=x)
        opt = Adadelta(lr=0.00025)
        # opt = Adam(lr=0.0001)
        model.compile(loss='mse', optimizer=opt)
        model.summary()
        return model

    def __train_model(self):

        stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        data = self.__load_data(self.__friends_avg)

        X_train_users, X_train_pois, X_train_category, X_train_friends, y_train = data[0]
        X_val_users, X_val_pois, X_val_category, X_val_friends, y_val = data[1]
        X_test_users, X_test_pois, X_test_category, X_test_friends, y_test = data[2]

        print("user_poi_category_friends_train shape: " + "(" + str(X_train_users[0].shape) + ", " + str(
            X_train_pois[0].shape) + ", " +
              str(X_train_friends[0].shape) + ")")
        print("user_poi_category_friends_val shape: " + "(" + str(X_val_users[0].shape) + ", " + str(
            X_val_pois[0].shape) + ", " +
              str(X_val_friends[0].shape) + ")")
        print("y_train, y_val shape:" + str(y_train.shape) + ", " + str(y_val.shape))

        if os.path.isfile('/content/drive/My Drive/WS2020/model_gw_category_mean') or os.path.isdir(
                '/content/drive/My Drive/WS2020/model_gw_category_mean'):
            print("PRE-TRAINED MODEL LOADED")
            model = load_model('/content/drive/My Drive/WS2020/model_gw_category_mean')
            self.__computeIndexes()
            self.__evaluate(model, data, self.__friends_avg)
        else:

            model = self.__RecommenderNet(self.__n_users, self.__users.getPOIcount(), 10, X_train_users[0].shape,
                                          X_train_pois[0].shape, X_train_category[0].shape, X_train_friends[0].shape)

            result = model.fit([X_train_users, X_train_pois, X_train_category, X_train_friends], y_train,
                               batch_size=1024, epochs=10,
                               callbacks=[stopping],
                               validation_data=([X_val_users, X_val_pois, X_val_category, X_val_friends], y_val),
                               verbose=1)

            model.save('/content/drive/My Drive/WS2020/model_gw_category_mean')

    def __evaluate(self, model, data, friends_avg):  # k1 = 10, k2 = 5

        k, k1 = 10, 5
        X_test_users, X_test_pois, X_test_category, X_test_friends, y_test = data[2]
        X = np.array(self.__all_data[:, : 3 + friends_avg])
        all_count = 0
        hit_count = 0
        step = 1
        total_accuracy = 0
        all_interests = self.__POI_ids
        print(list(X_test_users[:10]))
        for user_id in list(X_test_users):

            user_interests = list(self.__users.getVisitorPOIs(user_id[0]))
            if k < len(user_interests):
                removed_k = random.sample(user_interests, k=k)
            else:
                removed_k = user_interests
            if len(removed_k) > 0:
                # removed_k = filter(lambda a: a in to_remove, user_interests)
                difference = list(set(all_interests) - set(removed_k))
                not_visited_k = random.sample(difference, k=k)
                all_k = set(removed_k).union(set(not_visited_k))
                print()
                # print("all_K: "+str(all_k))
                recommended = []
                # print(X)
                # user_index = self.__indexes_dict[user_id]
                user_ID = np.reshape(np.array([user_id]), (1, 1))
                # user_friends = X[(X[:, 0] == user_id)]
                for poi_ID in all_k:
                    user_friends = X[(X[:, 0] == user_id) & (X[:, 1] == poi_ID)]
                    # user_friends = user_friends[(user_friends[:, 1] == poi_ID)]
                    if len(user_friends) == 0:
                        user_friends = np.zeros(shape=(1, 1))
                    else:
                        print("############################ user_friends len > 0: " + str(
                            user_friends) + " ########################################")
                        user_friends = user_friends[0][3]
                        user_friends = np.reshape(user_friends, (1, 1))
                    poi_id = np.reshape(np.array([poi_ID]), (1, 1))
                    category_name = self.__POI_category[poi_ID]
                    category_id = np.reshape(np.array(self.__category_to_id[category_name]), (1, 1))
                    # print("user_id: "+str(user_ID)+", poi_id: "+str(poi_id)+", friends: "+str(user_friends))
                    prediction = model.predict([user_ID, poi_id, category_id, user_friends])
                    recommended.append((poi_ID, prediction))

                result = sorted(recommended, key=lambda x: x[1], reverse=True)
                # print("Recommended POIs: "+str(result))
                # print("All results: "+str(result))
                result = [x[0] for x in result[:k1]]
                print("Recommended POIs: " + str(result))
                print("Original POIs: " + str(removed_k))
                hit_count = len(np.intersect1d(result, removed_k))
                den = min(len(result), len(removed_k))
                if den > 0:
                    user_accuracy = hit_count / den
                    total_accuracy += user_accuracy
                    print("Accuracy for user " + str(user_id) + " :" + str(user_accuracy * 100))
                    print("[Step " + str(step) + "] Partial global accuracy: " + str(total_accuracy / step))
                    print("------------------------------------------------------------")
                    step += 1

    def __computeIndexes(self):
        map = self.__users.getMap()
        indexes_dict = dict()
        start = 0
        for user_id in self.__users.getVisitorsIDs():
            indexes_dict[user_id] = start
            start += len(map[user_id].keys())
        self.__indexes_dict = indexes_dict

