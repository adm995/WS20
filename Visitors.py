import gzip
class Visitors:
    def __init__(self, filename):
        self.__filename = filename
        self.__dic = dict()  #<int, <string, float> >
        self.__POIcount = 0
        self.__mapper = dict()  # maps the sequential ID to the original ones
        self.__readData(self.__filename)



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
        locDic = dict()
        locCount = 0
        visID = None
        visCount = 0
        with gzip.open(filename, 'rt') as fp:
            line = fp.readline()
            while line:
                line = fp.readline()
                strings = line.split()
                if len(strings) == 5:
                    visID = int(strings[0])
                    locString = strings[4]
                    if locString not in locDic:
                        locDic[locString] = locCount
                        locCount += 1
                    locID = locDic[locString]
                    # Update the Map entries for the current visitor
                    if visCount in self.__dic:
                        tmpMap = self.__dic[visCount]
                        if locID in tmpMap:
                            tmpCts = tmpMap.get(locID)
                            self.__dic[visCount][locID] = tmpCts+1
                        else:
                            self.__dic[visCount][locID] = float(1)
                    else:
                        self.__dic[visCount] = dict()
                        self.__mapper[visCount] = visID
                        self.__dic[visCount][locID] = float(1)
                        visCount += 1
            self.__POIcount = locCount
            fp.close()

    def getVisitorPOIs(self, visID):
        """
        Given a visitor ID returns its set of visited POI.
        :param visID: a visitor's ID.
        :return: The set of POI IDs that represents all the visited POI by the visitor specified by id visID.
        """
        return self.__dic[visID].keys()

    def getVisitorCts(self, visID, locID):
        """
        :param visID: the ID of a specific user.
        :param locID: the ID of a specific location.
        :return: The number of times in which the visitor with id visID has been at POI with id locID.
        """
        if visID not in self.getVisitorsIDs():
            print("This visitor is not in the dataset")
            return
        if self.__dic[visID][locID] is None:
            return 0
        return self.__dic[visID][locID]


    def getVisitorsIDs(self):
        """
        The set of visitors ids in the dataset.
        :return: The sorted set of visitors ids.
        """
        sortedKeys = list(self.__dic.keys())
        sortedKeys.sort()
        return sortedKeys

    def getMap(self):
        """
        :return: The entire dictionary of users.
        """
        return self.__dic

    def getPOIcount(self):
        """
        :return: The number of distinct POI in the dataset.
        """
        return self.__POIcount

    def getMapper(self):
        """
        :return: A dictionary that maps the sequential ID given to visitors to the original one in the dataset.
        """
        return self.__mapper
