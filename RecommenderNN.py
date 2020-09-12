import networkx as nx
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Embedding, Dense, Dropout, Concatenate, Input, Activation, Lambda, Flatten
from keras.optimizers import Adadelta
from keras.models import Model
import random
from keras import callbacks
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
from sklearn.preprocessing import MinMaxScaler
import os
from keras.models import load_model
from UsersReader import UsersReader


class RecommenderNN:

    def __init__(self, users: UsersReader, G: nx.Graph, friends_avg: float, is_full: bool, input_dir: str):
        """
        This class provides all the methods to train and evaluate the recommender system.
        :param users: Object containing all the informations about users, categories and friendhsip
        :param G: Friendship graph
        :param friends_avg: Avg number og freind per user
        :param is_full: Flag that is true if the stored version of the model is the 'Full' one
        :param input_dir: input directory
        """
        self.__users = users
        self.__input = input_dir
        self.__friends_avg = round(friends_avg)
        self.__is_full = is_full
        self.__POI_count = users.getPOIcount()
        self.__POI_ids = users.getPOIsIDs()
        self.__category_to_id = dict()
        self.__G = G

        ind = 0
        self.__POI_category = users.getCategories()
        for cat in set(self.__POI_category.values()):
            self.__category_to_id[cat] = ind
            ind += 1

        self.__mapper = users.getMapper("toID")
        input_path = os.path.join(self.__input, "dataset_small_random.csv")
        if os.path.isfile(input_path):
            self.__all_data = loadtxt(input_path, delimiter=',')
        else:
            self.__all_data = self.__preprocess(users, G)
        self.__n_users = int(max(self.__all_data[:, 0]))
        self.__train_model()

    def __preprocess(self, users: UsersReader, g: nx.Graph):
        """
        Merge in a single matrix the data from each user and its interests with the informations about the friendship
        from the friendship graph.
        :param users: Users read from dataset
        :param g: friendship graph
        :return: input matrix for the neural network
        """
        total_visits = []
        ids = users.getVisitorsIDs()
        for user_id in ids:
            total_visits.append(len(users.getVisitorPOIs(user_id)))
        rows_number = sum(total_visits)
        cols_number = 4 + self.__friends_avg  # user_id, poi_id, poi_category, friends_cts, user_cts
        all_data = np.zeros((rows_number, cols_number))
        row = 0
        for user_id in ids:
            print(user_id)
            friends_cts = []
            for poi_id in users.getVisitorPOIs(user_id):
                all_data[row, 0] = self.__mapper[str(user_id)]
                all_data[row, 1] = poi_id
                category_name = self.__POI_category[poi_id]
                all_data[row, 2] = self.__category_to_id[category_name]
                user_friends = set(g[user_id].keys())
                common_poi_friends = user_friends.intersection(users.getPOIvisitors(poi_id))

                for friend_id in user_friends:
                    friends_cts.append(users.getVisitorCts(friend_id, poi_id))
                if self.__is_full:
                    for col in range(3, cols_number):
                        if len(common_poi_friends) > 0:
                            common_poi_friends = list(common_poi_friends)
                            all_data[row, col] = common_poi_friends.pop(random.randint(0, len(common_poi_friends) - 1))
                        else:
                            all_data[row, col] = 0
                    all_data[row, -1] = users.getVisitorCts(user_id, poi_id)
                    row += 1
                else:
                    for friend_id in common_poi_friends:
                        friends_cts.append(users.getVisitorCts(friend_id, poi_id))
                    avg_friends_rating = 0 if len(friends_cts) == 0 else (sum(friends_cts) / len(friends_cts))
                    all_data[row, 3] = avg_friends_rating
                    all_data[row, -1] = users.getVisitorCts(user_id, poi_id)
                    row += 1

        data = asarray(all_data)
        input_path = os.path.join(self.__input, "dataset_small_random.csv")
        savetxt(input_path, data, delimiter=',')

        return all_data

    def __load_data(self):
        """
        Split the dataset into train, test and validation sets.
        :return:
        """
        sc = MinMaxScaler()
        X = np.array(self.__all_data[:, : 3 + self.__friends_avg])

        y = np.array(self.__all_data[:, 3 + self.__friends_avg])
        y = y.reshape(-1, 1)
        y = sc.fit_transform(y)

        data = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, test_size=0.4, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

        X_train_users = np.array(X_train[:, 0])
        X_train_pois = np.array(X_train[:, 1])
        X_train_category = np.array(X_train[:, 2])
        X_train_friends = np.array(X_train[:, 3:])
        train_samples = X_train_users.shape[0]

        X_train_users = np.reshape(X_train_users, (train_samples, 1))
        X_train_pois = np.reshape(X_train_pois, (train_samples, 1))
        X_train_category = np.reshape(X_train_category, (train_samples, 1))
        X_train_friends = np.reshape(X_train_friends, (train_samples, self.__friends_avg))
        y_train = np.reshape(y_train, (train_samples, 1))
        data.append((X_train_users, X_train_pois, X_train_category, X_train_friends, y_train))

        X_val_users = np.array(X_val[:, 0])
        X_val_pois = np.array(X_val[:, 1])
        X_val_category = np.array(X_val[:, 2])
        X_val_friends = np.array(X_val[:, 3:])
        val_samples = X_val_users.shape[0]

        X_val_users = np.reshape(X_val_users, (val_samples, 1))
        X_val_pois = np.reshape(X_val_pois, (val_samples, 1))
        X_val_category = np.reshape(X_val_category, (val_samples, 1))
        X_val_friends = np.reshape(X_val_friends, (val_samples, self.__friends_avg))
        y_val = np.reshape(y_val, (val_samples, 1))
        data.append((X_val_users, X_val_pois, X_val_category, X_val_friends, y_val))

        X_test_users = np.array(X_test[:, 0])
        X_test_pois = np.array(X_test[:, 1])
        X_test_category = np.array(X_test[:, 2])
        X_test_friends = np.array(X_test[:, 3:])
        test_samples = X_test_users.shape[0]

        X_test_users = np.reshape(X_test_users, (test_samples, 1))
        X_test_pois = np.reshape(X_test_pois, (test_samples, 1))
        X_test_category = np.reshape(X_test_category, (test_samples, 1))
        X_test_friends = np.reshape(X_test_friends, (test_samples, self.__friends_avg))
        y_test = np.reshape(y_test, (test_samples, 1))
        data.append((X_test_users, X_test_pois, X_test_category, X_test_friends, y_test))

        return data

    def __RecommenderNet(self, n_users, n_pois, n_categories, n_factors, min_rating, max_rating, user_shape, poi_shape,
                         category_shape, friends_shape) -> Model:
        """
        Return the neural network model ready to be trained on processed input data.
        :param n_users: total number of users in the dataset
        :param n_pois: total number of POIs in the dataset
        :param n_categories: Total number of categories in the dataset
        :param n_factors: number of latent factors
        :param min_rating: min cts
        :param max_rating: max cts
        :param user_shape: the array shape of a user sample
        :param poi_shape: the array shape of a poi sample
        :param category_shape: the array shape of a category sample
        :param friends_shape: the array shape of a friend sample
        :return: NN model
        """

        user = Input(shape=user_shape, name="User_ID")
        user_embedding = Embedding(n_users + 1, n_factors, embeddings_initializer='he_normal')(user)
        user_vec = Flatten(name='FlattenUsers')(user_embedding)

        poi = Input(shape=poi_shape, name="poi_ID")
        poi_embedding = Embedding(n_pois + 1, n_factors, embeddings_initializer='he_normal')(poi)
        poi_vec = Flatten(name='FlattenPois')(poi_embedding)

        friends = Input(shape=friends_shape, name="friends")
        friends_vec = Flatten(name='FlattenFriends')(friends)

        category = Input(shape=category_shape, name="category")
        category_embedding = Embedding(n_categories + 1, n_factors, embeddings_initializer='he_normal')(category)
        category_vec = Flatten(name='FlattenCategories')(category_embedding)

        x = Concatenate(axis=1)([user_vec, poi_vec, category_vec, friends_vec])
        x = Dropout(0.05)(x)

        x = Dense(20, kernel_initializer='he_uniform')(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(1, kernel_initializer='he_uniform')(x)
        x = Activation('sigmoid')(x)
        x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)
        model = Model(inputs=[user, poi, category, friends], outputs=x)
        opt = Adadelta(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)
        model.summary()
        return model

    def __train_model(self):
        """
        Train the model locally stored or create, train and save a new model from skretch
        :return:
        """
        stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        data = self.__load_data()

        X_train_users, X_train_pois, X_train_category, X_train_friends, y_train = data[0]
        X_val_users, X_val_pois, X_val_category, X_val_friends, y_val = data[1]
        X_test_users, X_test_pois, X_test_category, X_test_friends, y_test = data[2]

        model_path = os.path.join(self.__input, "model_small_random")
        if os.path.isfile(model_path) or os.path.isdir(model_path):
            model = load_model(model_path)
            self.__evaluateK(model, data)
        else:
            model = self.__RecommenderNet(self.__n_users, self.__users.getPOIcount(), len(self.__POI_category), 8, 0, 1,
                                          X_train_users[0].shape,
                                          X_train_pois[0].shape, X_train_category[0].shape, X_train_friends[0].shape)

            result = model.fit([X_train_users, X_train_pois, X_train_category, X_train_friends], y_train, batch_size=64,
                               epochs=10,
                               callbacks=[stopping],
                               validation_data=([X_val_users, X_val_pois, X_val_category, X_val_friends], y_val),
                               verbose=2)

            model.save(model_path)
            self.__evaluateK(model, data)

    def __evaluateK(self, model, data):
        """
        Execute the entire evaluation process on the given model printing the partial accuracy of each iteration and
        the avg accuracy.
        :param model: An already trained NN model
        :param data: Input matrix used to train the model
        :return:
        """
        k = 10
        accuracies = dict()
        for k1 in range(k):
            accuracies[k1] = self.__evaluate(model, data, k, k1 + 1)

        for k1 in accuracies:
            print("Accuracy with k' = " + str(k1 + 1) + " recommended POIs is: " + str(accuracies[k1]))
        accuracies_list = list(accuracies.values())
        print("Avg Accuracy = " + str(sum(accuracies_list)/len(accuracies_list)))

    def __evaluate(self, model, data, k=10, k1=1) -> float:
        """
        Helper function that performs a single evaluation iteration: remove k interest from each user interests,
        select other k random POIs never seen by the user putting all together in a 2k list. The model recommends
        the best k' POIs among the 2k. The percentage of these k' that were actually part of the first removed k
        interests is the accuracy of the model for the the single user.
        :param model: NN model
        :param data: Input matrix used to train the model
        :param k: Number of user's interest to remove
        :param k1: number of POI to recommend from the 2k set
        :return: The accuracy value as a float number
        """
        X_test_users, X_test_pois, X_test_category, X_test_friends, y_test = data[2]
        X = np.array(self.__all_data[:, : 3 + self.__friends_avg])
        step = 1
        total_accuracy = 0
        all_interests = self.__POI_ids

        ids = random.sample(list(X_test_users), k=2000)
        for user_id in ids:
            user_id = user_id[0]
            user_interests = list(self.__users.getVisitorPOIs(user_id))
            if k < len(user_interests):
                visited_k = random.sample(user_interests, k=k)
            else:
                visited_k = user_interests
            if len(visited_k) > 0:
                difference = list(set(all_interests) - set(visited_k))
                not_visited_k = set(random.sample(difference, k=len(visited_k)))
                all_k = set(visited_k).union(set(not_visited_k))
                recommended = []
                user_ID = np.reshape(np.array([user_id]), (1, 1))
                input_friends = None
                category_id = None
                user_friends = None
                for poi_ID in all_k:
                    if poi_ID in not_visited_k:
                        user_friends = set(self.__G[user_id].keys())
                        common_int_friends = user_friends.intersection(self.__users.getPOIvisitors(poi_ID))
                        input_friends = np.zeros(shape=(1, self.__friends_avg))
                        category_name = self.__POI_category[poi_ID]
                        category_id = np.reshape(np.array(self.__category_to_id[category_name]), (1, 1))
                        for col in range(self.__friends_avg):
                            if len(common_int_friends) > 0:
                                common_int_friends = list(common_int_friends)
                                input_friends[0, col] = common_int_friends.pop(
                                    random.randint(0, len(common_int_friends) - 1))
                            else:
                                input_friends[0, col] = 0
                    else:
                        user_friends = X[(X[:, 0] == user_id)]
                        user_friends = user_friends[(user_friends[:, 1] == poi_ID)]
                        if len(user_friends) == 0:
                            category_name = self.__POI_category[poi_ID]
                            category_id = np.reshape(np.array(self.__category_to_id[category_name]), (1, 1))
                            input_friends = np.zeros(shape=(1, self.__friends_avg))
                        else:
                            category_id = np.reshape(user_friends[0][2], (1, 1))
                            input_friends = user_friends[0][3:]
                            input_friends = np.reshape(input_friends, (1, self.__friends_avg))
                    poi_id = np.reshape(np.array([poi_ID]), (1, 1))

                    if self.__is_full:
                        input_friends = np.average(input_friends)  # axis = 1
                        input_friends = np.reshape(input_friends, (1, self.__friends_avg))
                    prediction = model.predict([user_ID, poi_id, category_id, input_friends])
                    recommended.append((poi_ID, prediction))
                tmp_res = sorted(recommended, key=lambda x: x[1], reverse=True)
                result = [x[0] for x in tmp_res[:k1]]
                print("Recommended POIs: " + str(result))
                print("Original POIs: " + str(visited_k))
                hit_count = len(np.intersect1d(result, visited_k))
                den = min(len(result), len(not_visited_k))
                if den > 0:
                    user_accuracy = hit_count / den
                    total_accuracy += user_accuracy
                    print("Accuracy for user " + str(user_id) + " :" + str(
                        user_accuracy * 100) + "%,  with friends: " + str(input_friends))
                    print("Global accuracy: " + str(total_accuracy / step))
                    print("--------------")
                    print()
                    step += 1
        return total_accuracy / step
