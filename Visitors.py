import gzip
class Visitors:
    def __init__(self, filename):
        self.__filename = filename
        self.__dic = dict()  #<int, <string, float> >
        self.__readData(self.__filename)


    """ example of dataset row:
    visID                 ts                   L1           L2          PoI              ID
    0          2010 - 10 - 17T01: 48:53Z   39.747652     -104.99251      88      c46bf20db295831bd2d1718ad7e6f5

    Given the filename of the dataset this method reads each line from the dataset file and build a dictionary data - structure
    that for each visitor v stores all the visited POIs of v and the relative cts (the number of times that v has visited the POI).
    Are omitted all the PoI not visited from v, is implicitly assumed a 0 value for each of them.

    Example:
                    PoI x: -> 3
        user v ->   PoI y: -> 4
                    ...
                    PoI z: -> 8

                    PoI w: -> 1
        user u ->   PoI x: -> 3 
                    ...
                    PoI y: -> 3
    """
    def __readData(self, filename):
        locDic = dict()
        locCount = 0
        visID = None
        with gzip.open(filename,'rt') as fp:
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
                    locID = locDic.get(locString);
                    # Update the Map entries for the current user
                    if visID in self.__dic:
                        tmpMap = self.__dic[visID]
                        if locID in tmpMap:
                            tmpCts = tmpMap.get(locID)
                            self.__dic[visID][locID] = tmpCts+1
                        else:
                            self.__dic[visID][locID] = float(1)
                    else:
                        self.__dic[visID] = dict()
                        self.__dic[visID][locID] = float(1)
            fp.close()

    """ Given a visitor ID returns its set of visited POI"""
    def getVisitorPOIs(self, visID):
        return self.__dic[visID].keys()

    """Given the visitor ID and a POI 's ID returns how many time the visitor has visited that POI."""
    def getVisitorCts(self, visID, locID):
        if self.__dic[visID][locID] is None:
            return 0
        return self.__dic[visID][locID]

    """ Returns all visitors ids in the dataset. """
    def getVisitorsIDs(self):
        sortedKeys = list(self.__dic.keys())
        sortedKeys.sort()
        return sortedKeys

    """"Returns the entire data - structure."""
    def getMap(self):
        return self.__dic

