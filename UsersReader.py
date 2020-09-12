import gzip
from typing import Dict, List, Set
import networkx as nx
from Similarities import Similarities


class UsersReader:

    def __init__(self, filename: str, G: nx.Graph):
        """
        This class provides methods to read from the given filename the dataset extracting and putting in helper
        data structures all the important informations about users and POIs. So that they can be used in all the next
        steps of the project.
        :param filename: Where is stored the checkins dataset
        :param G: Friendship graph
        """
        self.__G = G
        self.__filename = filename
        self.__dic = dict()  # <int, <string, float> >
        self.__POI_count = 0
        self.__mapper_to_ID = dict()
        self.__mapper_to_str = dict()
        self.__POI_dict = dict()
        self.__POI_category = dict()
        self.__readData(self.__filename)
        sortedKeys = list(self.__dic.keys())
        sortedKeys.sort()
        self.__IDs = sortedKeys
        # self.__checkFriendsSim()

    def __readData(self, filename: str):
        """
            Given the filename of the dataset this method reads each line from the dataset file and build a dictionary
            data - structure that for each visitor v stores all the visited POIs of v and the relative cts (the number
            of times that v has visited the POI). Are omitted all the PoI not visited from v, is implicitly assumed a 0
            value for each of them. Moreover saves also informations about the POIs, creating a dictionary that maps
            each POI ID into the relative category.

            Example:
                            PoI x: -> 3
                user v ->   PoI y: -> 4
                            ...
                            PoI z: -> 8

                            PoI w: -> 1
                user u ->   PoI x: -> 3
                            ...
                            PoI y: -> 3
                :param filename:
                :return:
        """
        loc_dict = dict()
        vis_dict = dict()
        loc_count = 0
        vis_count = 0
        with gzip.open(filename, 'rt') as fp:
            line = fp.readline()
            while line:
                line = fp.readline()
                strings = line.split()
                if len(strings) == 3:
                    vis_string = strings[0]
                    vis_int = int(vis_string)
                    locString = strings[1]
                    category = strings[2]
                    if locString not in loc_dict:
                        loc_dict[locString] = loc_count
                        loc_count = loc_count + 1
                    loc_ID = loc_dict[locString]
                    if vis_string not in vis_dict:
                        vis_dict[vis_string] = vis_count
                        vis_count = vis_count + 1
                    vis_ID = vis_dict[vis_string]
                    # Update the Map entries for the current visitor
                    if vis_int in self.__dic:
                        tmp_map = self.__dic[vis_int]
                        if loc_ID in tmp_map:
                            tmp_cts = tmp_map[loc_ID]
                            self.__dic[vis_int][loc_ID] = tmp_cts + 1
                        else:
                            self.__dic[vis_int][loc_ID] = float(1)
                            if loc_ID in self.__POI_dict:
                                self.__POI_dict[loc_ID].add(vis_int)
                            else:
                                self.__POI_dict[loc_ID] = set()
                                self.__POI_dict[loc_ID].add(vis_int)
                                self.__POI_category[loc_ID] = category
                    else:
                        self.__dic[vis_int] = dict()
                        self.__dic[vis_int][loc_ID] = float(1)
                        if loc_ID not in self.__POI_category:
                            self.__POI_dict[loc_ID] = set()
                            self.__POI_category[loc_ID] = category
                        self.__POI_dict[loc_ID].add(vis_int)
                        self.__mapper_to_ID[vis_string] = vis_ID
                        self.__mapper_to_str[vis_ID] = vis_string
        self.__POI_count = loc_count
        fp.close()

    def getVisitorPOIs(self, user_id):
        """
        Given a visitor ID returns its set of visited POI.
        :param user_id: a visitor's ID.
        :return: The set of POI IDs that represents all the visited POI by the visitor specified by id visID.
        """
        if user_id not in self.__dic:
            return []
        if self.__dic[user_id] is None:
            return []
        return self.__dic[user_id].keys()

    def getVisitorCts(self, user_id, loc_id):
        """
        :param user_id: the ID of a specific user.
        :param loc_id: the ID of a specific location.
        :return: The number of times in which the visitor with id vis_id has been at POI with id loc_id.
        """
        if user_id not in self.__dic:
            return 0
        if loc_id not in self.__dic[user_id]:
            return 0
        return self.__dic[user_id][loc_id]

    def getVisitorsIDs(self) -> List[int]:
        """
        The set of visitors ID in the dataset.
        :return: The sorted set of visitors ids.
        """
        return self.__IDs

    def getMap(self) -> Dict:
        """
        :return: The entire dictionary of users.
        """
        return self.__dic

    def getPOIcount(self) -> int:
        """
        :return: The number of distinct POI in the dataset.
        """
        return self.__POI_count

    def getMapper(self, string):
        """
        :param string:  A string in {"toID, toStr"} to specify which mapper to return.
        :return: the mapper (if any) associated to the input string.
        """
        if string not in {"toID", "toStr"}:
            raise ValueError
        if string == "toID":
            return self.__mapper_to_ID
        else:
            return self.__mapper_to_str

    def getCategories(self):
        """
        :return: A dictionary that maps each POI's ID into its category string.
        """
        return self.__POI_category

    def getPOICategory(self, poi_id):
        """
        :return: A string that represents the category associated to the given POI id.
        """
        return self.__POI_category[poi_id]

    def getPOIvisitors(self, poi_id) -> Set[int]:
        """
        Given a POI id the methods returnss the IDs set of all users that visited that POI.
        :param poi_id: the ID of a location
        :return: The set of users that have visited the POI with ID poi_id
        """
        return self.__POI_dict[poi_id]

    def getPOIsIDs(self):
        """
        Return the IDs of all users in the dataset.
        :return:
        """
        return self.__POI_dict.keys()

    def __checkFriendsSim(self):
        """
        Compute the average Jaccard similarity between the vectors of visited POI for each pair of users
        :return:
        """
        all = 1
        global_sim = 0
        users = list(self.__IDs)
        for i in range(len(users)):
            user = users[i]
            for friend in self.__G[user]:
                if friend in self.__dic:
                    all += 1
                    sim = Similarities.get_jaccard_similarity(self.__dic[user], self.__dic[int(friend)])
                    global_sim += sim
            if all > 0:
                print("Mean friends similarity: "+str(global_sim/all))
