# NOAA SEALION COUNTING CHALLENGE COMPETITION CODE 58th/385
# Author : Young-chul Yoon

import numpy as np
import pandas as pd
import cv2
import os
import random
import csv

abs_path = os.path.dirname(os.path.realpath(__file__))
path = abs_path + "/"

# Check whether folder_path exist or not and make directory if it doenst exist
def folder_check(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# background_extraction
def background(folder_path, image, false_count):
    folder_check(folder_path)
    local_num = 0
    # 30-background randomly
    while local_num <= 30:
        x = random.randrange(40, len(image[0]) - 40)
        y = random.randrange(40, len(image) - 40)
        crop = image[int(y) - 40:int(y) + 40, int(x) - 40:int(x) + 40]
        false_count = false_count + 1
        local_num = local_num + 1
        cv2.imwrite(os.path.join(folder_path, str(false_count) + ".jpg"), crop)
    # 70-background randomly with no black part
    while local_num <= 100:
        x = random.randrange(40, len(image[0]) - 40)
        y = random.randrange(40, len(image) - 40)
        crop = image[int(y) - 40:int(y) + 40, int(x) - 40:int(x) + 40]
        if np.amin(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) != 0:
            false_count = false_count + 1
            local_num = local_num + 1
            cv2.imwrite(os.path.join(folder_path, str(false_count) + ".jpg"), crop)
    return false_count

def false_generator(folder_path, cut, land_count, location):
    folder_check(folder_path)
    #print(len(location))
    for i in range(len(location)):
        if (location[i][0] - 120 < 0 or location[i][0] + 120 > len(cut[0]) or location[i][1] - 120 < 0 or location[i][
            1] + 120 > len(cut)):
            continue
        x = location[i][0]
        y = location[i][1]

        direction = [[1, 1], [1, 0], [1, -1], [0, 1], [0, -1], [-1, 1], [-1, 0], [-1, -1]]
        for pick in range(8):
            p_dir = direction[pick]
            x_base = x - 80 * p_dir[0]
            y_base = y - 80 * p_dir[1]

            land = cut[int(y_base) - 40:int(y_base) + 40, int(x_base) - 40:int(x_base) + 40]
            if np.amin(cv2.cvtColor(land, cv2.COLOR_BGR2GRAY)) != 0:
                land_count = land_count + 1
                cv2.imwrite(folder_path + str(land_count) + ".jpg", land)
    return land_count

def read_coord(img_num):
    f = open(path + "input/coords.csv")
    reader = csv.reader(f)
    arr = []
    for row in reader:
        if len(row) != 0:
            if int(row[0]) == img_num and int(row[1]) <= 3:
                arr.append([row[2],row[3]])
            elif int(row[0]) > img_num:
                break
    return arr

def read_mismatch(test_train):
    f = open(path + "input/MismatchedTrainImages.txt", 'r')
    sav_arr = []
    f.readline()
    for next in f.readlines():
        if(test_train) == "test_set":
            if int(next.replace("\n", "")) < 750:
                continue
        sav_arr.append(next.replace("\n", ""))
    return sav_arr

def get_coord(img_num):
    f = open(path + "input/coords.csv")
    reader = csv.reader(f)
    arr = []
    dump = next(reader)
    for row in reader:
        if len(row) != 0:
            if int(row[0]) == img_num and int(row[1]) <= 3:
                arr.append([row[1],row[2],row[3]])
            elif int(row[0]) > img_num:
                break
    return arr

def pos_neg_generator(file_names, classes, test_train, concat):
    if test_train == "test_set":
        start_num = 750
        end_num = 948
        default_path = "input/test_set/"
    elif test_train == "train_set":
        start_num = 0
        end_num = 750
        default_path = "input/train_set/"
    elif test_train == "full":
        start_num = 0
        end_num = 948
        default_path = "input/full_set/"

    dotted_img_path = "input/TrainDotted/"
    train_img_path = "input/Train/"

    folder_check(default_path)

    file_names = sorted(file_names,
                        key=lambda item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))
    file_names = file_names[start_num:end_num]

    global_count = [0, 0, 0, 0, 0]
    concat_count = 0
    false_count = 0
    land_count = 0
    count_df = pd.DataFrame(index=file_names, columns=classes).fillna(0)

    mis_arr = read_mismatch(test_train)
    folder_check(default_path + "blob_point/")

    for filename in file_names:
        print(filename)
        if len(mis_arr) > 0:
            if mis_arr[0] == filename.replace('.jpg', ""):
                print("Mismatch file found")
                mis_arr.remove(mis_arr[0])
                continue

        arr = get_coord(int(filename.replace('.jpg', '')))
        cls = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]

        image = cv2.imread(train_img_path + filename)
        dot_img = cv2.imread(dotted_img_path + filename)
        location = []
        cut = np.copy(dot_img)
        print("len : ", len(arr))
        for i in range(len(arr)):
            print(i)
            cls_num = int(arr[i][0])
            y = int(arr[i][1])
            x = int(arr[i][2])
            if (y < 40 or y > image.shape[0] - 40 or x < 40 or x > image.shape[1] - 40):
                continue
            if concat == True:
                path = default_path + "sealions/"
                each_path = default_path + cls[cls_num] + "/"
            else:
                path = default_path + cls[cls_num]+"/"
            folder_check(path)
            folder_check(each_path)
            count_df[cls[cls_num]][filename] += 1
            global_count[cls_num] += 1
            crop_img = image[y - 40:y + 40, x - 40:x + 40]
            concat_count += 1
            if concat == True:
                cv2.imwrite(path + str(concat_count) + ".jpg", crop_img)
                cv2.imwrite(each_path + str(global_count[cls_num]) + ".jpg", crop_img)
            else:
                cv2.imwrite(path + str(global_count[cls_num]) + ".jpg", crop_img)
            location.append([x, y])
            cv2.rectangle(cut, (x - 40, y - 40), (x + 40, y + 40), 0, -1)
        false_count = background(default_path + "background/", cut, false_count)
        land_count = false_generator(default_path + "false_image/", cut, land_count, location)

if __name__ == "__main__":
    classes = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]
    file_names = os.listdir("input/TrainDotted/")
    #pos_neg_generator(file_names, classes, "full", True)
    pos_neg_generator(file_names, classes, "train_set", True)
    pos_neg_generator(file_names, classes, "test_set", True)