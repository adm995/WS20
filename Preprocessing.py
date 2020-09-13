import csv
import json
import os


class Preprocessing:

    def __init__(self, data_dir: str):
        """
        This class provides methods to preprocess the files from Gowalla datasets deleting useless informations and
        creating a more compact version including POIs categories. The methods needs files from
        <https://drive.google.com/file/d/0BzpKyxX1dqTYRTFVYTd1UG81ZXc/view> to be put into the data project folder.
        Anyway it is no more useful, all post-processing data are already in data folder.
        """
        self.__data_dir = data_dir
        self.__subcat_to_maincat = dict()  # maps the nested category name of a spot to the first layer category
        self.read_hierarchy_json()
        self.read_spots_csv()

    def read_spots_csv(self):
        """
        Read the spot subsets files from the full Gowalla dataset to extract the category of each POI and associates
        each POI ID to its relative category. Then modify the SNAP checkins dataset adding the extracted category
        information.
        :return:
        """

        file_subset1 = os.path.join(self.__data_dir, "gowalla_spots_subset1.csv")
        file_subset2 = os.path.join(self.__data_dir, "gowalla_spots_subset2.csv")
        dataset = os.path.join(self.__data_dir, "Gowalla_totalCheckins.txt")

        if os.path.isfile(file_subset1) and os.path.isfile(file_subset2) and os.path.isfile(dataset):
            poi_to_cat = dict()
            with open(file_subset1, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                next(csv_reader, None)
                for line in csv_reader:
                    poi_id = line[0]
                    sub_category_id = int(eval(line[-1])[0]["url"].replace("/categories/", ""))
                    if sub_category_id in self.__subcat_to_maincat:
                        poi_cat = self.__subcat_to_maincat[sub_category_id]
                        poi_to_cat[poi_id] = poi_cat
                    else:
                        poi_to_cat[poi_id] = 0

            with open(file_subset2, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                next(csv_reader, None)
                for line in csv_reader:
                    poi_id = line[0]
                    poi_to_cat[poi_id] = 0
            # checkins dataset: userid,placeid,datetime

            with open("gowalla_small_cat.txt", "a") as my_output_file:
                with open(dataset, "r") as my_input_file:
                    #csv_reader = csv.reader(my_input_file)
                    #next(csv_reader, None)
                    lines = my_input_file.readlines()
                    for line in lines:
                        strings = line.split()
                        user_id = strings[0]
                        poi_id = strings[-1]
                        if poi_id in poi_to_cat:
                            category = poi_to_cat[poi_id]
                        else:
                            category = 0
                        my_output_file.write(user_id + " " + poi_id + " " + str(category) + "\n")
                my_output_file.close()
        else:
            print("Please put in the project data folder all the missing files")

    def read_hierarchy_json(self):
        """
        Recursively read the hierarchy json extracting all the categories from root categories to first children
        layer categories.
        :return:
        """
        file = os.path.join(self.__data_dir, "gowalla_category_structure.json")
        if os.path.isfile(file):
            f = open(file)
            json_obj = json.load(f)["spot_categories"]
            self.__recursiveHelper(json_obj, None)
            for category in self.__subcat_to_maincat:
                if self.__subcat_to_maincat[category] is None:
                    self.__subcat_to_maincat[category] = category
                    break
        else:
            print("Please put in the project data folder all the missing files")

    def __recursiveHelper(self, json_obj, parent):
        for sub_category in json_obj:
            subcat_id = int(sub_category["url"].replace("/categories/", ""))
            self.__subcat_to_maincat[subcat_id] = parent
            if "spot_categories" in sub_category:
                self.__recursiveHelper(sub_category["spot_categories"], subcat_id)
