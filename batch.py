# NOAA SEALION COUNTING CHALLENGE COMPETITION CODE 58th/385
# Author : Young-chul Yoon

import numpy as np
import os
import random
from random import shuffle
from skimage import io


classes = ["adult_males", "subadult_males",  "adult_females", "juveniles", "background", "false_image"]
train_cls = ["sealions", "background", "false_image"]
second_train_cls = ["sealions", "background", "false_image", "ft_img"]

class_label = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
detection_label = [[0, 1], [1, 0]]

abs_path = os.path.dirname(os.path.realpath(__file__))
path = abs_path + "/input/"

# Generate whole list of samples
def list_generator(mode, model):

    if mode == "train":
        cls = train_cls
        new_default = path + "train_set/"
        if model == "second_detection":
            cls = second_train_cls
        elif model == "classification":
            cls = classes
    else:
        cls = classes
        new_default = path + "test_set/"

    sample_num = []
    for age in cls:
        group_file_names = os.listdir(new_default + age +"/")
        for file_name in group_file_names:
            new_set = file_name.split('.')
            if(len(new_set) < 2):
                group_file_names.remove(file_name)
            elif(new_set[1] != "jpg"):
                group_file_names.remove(file_name)
        group_file_names = sorted(group_file_names, key = lambda item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))
        #Get the number of samples of each class
        #The last element is Thumbs.db
        sample_num.append(int(group_file_names[-2].split('.')[0]))

    whole_batch = []
    for i in range(0, len(cls)):
        temp_list = list(range(1, sample_num[i]))
        whole_batch.append(temp_list)

    for i in range(0, len(cls)):
        shuffle(whole_batch[i])

    return whole_batch

# Pick samples from whole_list and create batch
def batch_generator(whole_list, model, mode):
    sample_batch = []
    label_batch = []

    if mode == "train":

        default_folder = path + "train_set/"

        if model == "classification":
            for i in range(10):
                # male
                pick = whole_list[0][0]
                whole_list[0].remove(pick)
                sample_batch.append(default_folder + classes[0] + "/" + str(pick) + ".jpg")
                label_batch.append(class_label[0])
                #subadult
            for i in range(10):
                pick = whole_list[1][0]
                whole_list[1].remove(pick)
                sample_batch.append(default_folder + classes[1] + "/" + str(pick) + ".jpg")
                label_batch.append(class_label[1])
                #female
            for i in range(50):
                pick = whole_list[2][0]
                whole_list[2].remove(pick)
                sample_batch.append(default_folder + classes[2] + "/" + str(pick) + ".jpg")
                label_batch.append(class_label[2])
                #juvenile
            for i in range(50):
                pick = whole_list[3][0]
                whole_list[3].remove(pick)
                sample_batch.append(default_folder + classes[3] + "/" + str(pick) + ".jpg")
                label_batch.append(class_label[3])
        elif model == "detection":
            #sealions*30
            for i in range(30):
                pos_pick = whole_list[0][0]
                whole_list[0].remove(pos_pick)
                sample_batch.append(default_folder + "sealions/" + str(pos_pick) + ".jpg")
                label_batch.append(detection_label[0])
            #background*30
            for i in range(30):
                neg_pick = whole_list[-2][0]
                whole_list[-2].remove(neg_pick)
                n_sample_str = default_folder + "background/" + str(neg_pick) + ".jpg"
                sample_batch.append(n_sample_str)
                label_batch.append(detection_label[1])
            #false_image*30
            for i in range(30):
                neg_pick = whole_list[-1][0]
                whole_list[-1].remove(neg_pick)
                n_sample_str = default_folder + "false_image/" + str(neg_pick) + ".jpg"
                sample_batch.append(n_sample_str)
                label_batch.append(detection_label[1])
        elif model == "second_detection":
            # sealions*20
            for i in range(30):
                pos_pick = whole_list[0][0]
                whole_list[0].remove(pos_pick)
                sample_batch.append(default_folder + "sealions/" + str(pos_pick) + ".jpg")
                label_batch.append(detection_label[0])
            # background*20
            for i in range(20):
                neg_pick = whole_list[1][0]
                whole_list[1].remove(neg_pick)
                n_sample_str = default_folder + "background/" + str(neg_pick) + ".jpg"
                sample_batch.append(n_sample_str)
                label_batch.append(detection_label[1])
            # false_image*20
            for i in range(20):
                neg_pick = whole_list[2][0]
                whole_list[2].remove(neg_pick)
                n_sample_str = default_folder + "false_image/" + str(neg_pick) + ".jpg"
                sample_batch.append(n_sample_str)
                label_batch.append(detection_label[1])
            # ft_img*20
            for i in range(15):
                neg_pick = whole_list[3][0]
                whole_list[3].remove(neg_pick)
                n_sample_str = default_folder + "ft_img/" + str(neg_pick) + ".jpg"
                sample_batch.append(n_sample_str)
                label_batch.append(detection_label[1])
    else:
        default_folder = "input/test_set/"

        #True-True accuracy check
        if model == "detection_sealion":
            class_pick = random.randrange(0, 4)
            pos_pick = whole_list[class_pick][0]
            whole_list[class_pick].remove(pos_pick)
            sample_batch.append(default_folder + classes[class_pick] + "/" + str(pos_pick) + ".jpg")
            label_batch.append(detection_label[0])
        #False-False accuracy check
        elif model == "detection_background":
            class_pick = random.randrange(4, 6)
            neg_pick = whole_list[class_pick][0]
            whole_list[class_pick].remove(neg_pick)
            sample_batch.append(default_folder + classes[class_pick] + "/" + str(neg_pick) + ".jpg")
            label_batch.append(detection_label[1])
        elif model == "classification":
            #male
            for i in range(10):
                class_pick = 0
                neg_pick = whole_list[class_pick][0]
                whole_list[class_pick].remove(neg_pick)
                sample_batch.append(default_folder + classes[class_pick] + "/" + str(neg_pick) + ".jpg")
                label_batch.append(class_label[class_pick])
            #subadult
            for i in range(10):
                class_pick = 1
                neg_pick = whole_list[class_pick][0]
                whole_list[class_pick].remove(neg_pick)
                sample_batch.append(default_folder + classes[class_pick] + "/" + str(neg_pick) + ".jpg")
                label_batch.append(class_label[class_pick])
            #female
            for i in range(10):
                class_pick = 2
                neg_pick = whole_list[class_pick][0]
                whole_list[class_pick].remove(neg_pick)
                sample_batch.append(default_folder + classes[class_pick] + "/" + str(neg_pick) + ".jpg")
                label_batch.append(class_label[class_pick])
            #juvenile
            for i in range(10):
                class_pick = 3
                neg_pick = whole_list[class_pick][0]
                whole_list[class_pick].remove(neg_pick)
                sample_batch.append(default_folder + classes[class_pick] + "/" + str(neg_pick) + ".jpg")
                label_batch.append(class_label[class_pick])
    for i in range(3):
        #Read image
        out_sample = io.imread(sample_batch[0])
        #out_sample = transform.resize(out_sample, (40, 40), mode='reflect')
        out_sample = np.array([out_sample])
        for i in range(1, len(sample_batch)):
            read_sample = io.imread(sample_batch[i])
            #read_sample = transform.resize(read_sample, (40, 40), mode='reflect')
            out_sample = np.append(out_sample, [read_sample], axis=0)

    return out_sample, label_batch, whole_list