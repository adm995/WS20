import gzip
from geopy.geocoders import Nominatim
from functools import partial
from typing import Dict, List, Set
from collections import Counter


class UsersReader:
    def __init__(self, filename):
        self.__filename = filename
        self.__dic = dict()  # <int, <string, float> >
        self.__POI_count = 0
        self.__mapper_to_ID = dict()
        self.__mapper_to_str = dict()
        self.__POI_dict = dict()
        self.__coordinates = dict()
        self.__readData(self.__filename)
        #print(self.__POI_dict)
        #print(self.__POI_dict[0])
        sortedKeys = list(self.__dic.keys())
        sortedKeys.sort()
        self.__IDs = sortedKeys
        #self.__max_cts = self.__getMaxCts()
        #self.__computePOIdict()
    """ example of dataset row:
    visID                 ts                   L1           L2          PoI              ID
    0          2010 - 10 - 17T01: 48:53Z   39.747652     -104.99251      88      c46bf20db295831bd2d1718ad7e6f5
    """
    def __readData(self, filename):
        """
            Given the filename of the dataset this method reads each line from the dataset file and build a dictionary
            data - structure that for each visitor v stores all the visited POIs of v and the relative cts (the number
            of times that v has visited the POI). Are omitted all the PoI not visited from v, is implicitly assumed a 0
            value for each of them.

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
                if len(strings) == 5:
                    vis_string = strings[0]
                    coords = (strings[2], strings[3])
                    vis_int = int(vis_string)
                    locString = strings[4]
                    if locString not in loc_dict:
                        loc_dict[locString] = loc_count
                        loc_count = loc_count + 1
                    loc_ID = loc_dict[locString]
                    if vis_string not in vis_dict:
                        vis_dict[vis_string] = vis_count
                        vis_count = vis_count + 1
                    vis_ID = vis_dict[vis_string]
                    # Update the Map entries for the current visitor
                    if vis_int in self.__dic.keys():
                        tmp_map = self.__dic[vis_int]
                        if loc_ID in tmp_map:
                            tmp_cts = tmp_map[loc_ID]
                            self.__dic[vis_int][loc_ID] = tmp_cts+1
                        else:
                            self.__dic[vis_int][loc_ID] = float(1)
                            if loc_ID in self.__POI_dict.keys():
                                self.__POI_dict[loc_ID].add(vis_int)
                            else:
                                self.__POI_dict[loc_ID] = set()
                                self.__POI_dict[loc_ID].add(vis_int)
                    else:
                        self.__dic[vis_int] = dict()
                        self.__dic[vis_int][loc_ID] = float(1)
                        self.__POI_dict[loc_ID] = set()
                        self.__POI_dict[loc_ID].add(vis_int)
                        if loc_ID not in self.__coordinates.keys():
                            self.__coordinates[loc_ID] = coords
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
        if user_id not in self.getVisitorsIDs():
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
        if user_id not in self.getVisitorsIDs():
            #print("This visitor is not in the dataset")
            return
        if self.__dic[user_id][loc_id] is None:
            return 0
        return self.__dic[user_id][loc_id]

    def getVisitorsIDs(self) -> List[int]:
        """
        The set of visitors ID in the dataset.
        :return: The sorted set of visitors ids.
        """
        return self.__IDs

    def getMap(self):
        """
        :return: The entire dictionary of users.
        """
        return self.__dic

    def getPOIcount(self):
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

    def getPOIsCoords(self):
        """
        :return: A dictionary that maps each POI's ID into its pair of coordinates.
        """
        return self.__coordinates

    def __getPOIsCategory(self):
        """
        :return: For each pair of POI's coordinates assign a category label exploiting GeoPy libraries.
        """
        geolocator = Nominatim(user_agent="WS2020")
        coords = self.getPOIsCoords()
        reverse = partial(geolocator.reverse, language="en")
        for poi in coords.keys():
            print(reverse(coords[poi]).raw["address"])

    def getPOIvisitors(self, poi_id) -> Set[int]:
        return self.__POI_dict[poi_id]

    def __getMaxCts(self):
        self.__max_cts = 0
        all_cts  = []
        for user_id in self.__IDs:
            values = self.__dic[user_id].values()
            m = max(values)
            all_cts.extend(values)
            if self.__max_cts < m:
                self.__max_cts = m
        print("max cts: "+str(self.__max_cts))
        print("top 10 most common cts: "+str(Counter(all_cts).most_common()[-10:]))