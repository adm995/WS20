import csv
import json


class Preprocessing:

    def __init__(self):
        self.__subcat_to_maincat = dict()  # maps the nested name of a spot to the first layer category
        self.read_json()
        self.read_csv()

    def read_csv(self):

        file = 'C:/Users/Angelo/Desktop/gowalla/gowalla_spots_subset1.csv'
        poi_to_cat = dict()
        with open(file, 'r') as csv_file:
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

        file = 'C:/Users/Angelo/Desktop/gowalla/gowalla_spots_subset2.csv'
        with open(file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None)
            for line in csv_reader:
                poi_id = line[0]
                poi_to_cat[poi_id] = 0
        # checkins dataset: userid,placeid,datetime
        dataset = 'C:/Users/Angelo/Desktop/gowalla/gowalla_checkins.csv'
        with open("gowalla_full_7cat.txt", "a") as my_output_file:
            with open(dataset, "r") as my_input_file:
                csv_reader = csv.reader(my_input_file)
                next(csv_reader, None)
                for line in csv_reader:
                    user_id = line[0]
                    poi_id = line[1]
                    if poi_id in poi_to_cat:
                        category = poi_to_cat[poi_id]
                    my_output_file.write(user_id + " " + poi_id + " " + str(category) + "\n")
            my_output_file.close()

    def read_json(self):
        file = 'C:/Users/Angelo/Desktop/gowalla/gowalla_category_structure.json'
        f = open(file)
        root = json.load(f)["spot_categories"]
        self.__recursiveRead(root, None)
        for category in self.__subcat_to_maincat:
            if self.__subcat_to_maincat[category] is not None:
                c = category
                while True:
                    cat = c
                    super_cat = self.__subcat_to_maincat[cat]
                    if super_cat is None:
                        self.__subcat_to_maincat[category] = cat
                        break
                    else:
                        c = super_cat
        print(self.__subcat_to_maincat)

    def __recursiveRead(self, json_obj, caller):
        for sub_category in json_obj:
            cat_name = sub_category["name"]
            subcat_id = int(sub_category["url"].replace("/categories/", ""))
            self.__subcat_to_maincat[subcat_id] = caller
            if "spot_categories" in sub_category:
                self.__recursiveRead(sub_category["spot_categories"], subcat_id)
