import gzip
class Visitors:
    def __init__(self, filename):
        self.__filename = filename
        self.__dic = dict()  #<int, <string, float> >
        self.__POIcount = 0
        self.__mapperToID = dict()
        self.__mapperToStr = dict()
        self.__visDic = dict()
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
        visDic = dict()
        locCount = 0
        visCount = 0
        visID = 0
        with gzip.open(filename, 'rt') as fp:
            line = fp.readline()
            while line:
                line = fp.readline()
                strings = line.split()
                if len(strings) == 5:
                    visString = strings[0]
                    visInt = int(visString)
                    locString = strings[4]
                    if locString not in locDic:
                        locDic[locString] = locCount
                        locCount = locCount + 1
                    locID = locDic[locString]
                    if visString not in visDic:
                        visDic[visString] = visCount
                        visCount = visCount + 1
                    visID = visDic[visString]
                    # Update the Map entries for the current visitor
                    if visInt in self.__dic.keys():
                        tmpMap = self.__dic[visInt]
                        if locID in tmpMap:
                            tmpCts = tmpMap[locID]
                            self.__dic[visInt][locID] = tmpCts+1
                        else:
                            self.__dic[visInt][locID] = float(1)
                    else:
                        self.__dic[visInt] = dict()
                        self.__dic[visInt][locID] = float(1)
                        self.__mapperToID[visString] = visID
                        self.__mapperToStr[visID] = visString
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

    def getMapper(self, string):
        """
        :param string:  A string in {"toID, toStr"} to specify which mapper to return.
        :return:
        """
        if string not in {"toID", "toStr"}:
            print("This dictionary doesn't exist")
        if string == "toID":
            return self.__mapperToID
        else:
            return self.__mapperToStr
