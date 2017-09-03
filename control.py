# NOAA SEALION COUNTING CHALLENGE COMPETITION CODE 58th/385
# Author : Young-chul Yoon

from __future__ import print_function
import cnn
import batch
from skimage import io
from skimage import transform
import tensorflow as tf
import sliding_window
import extract
import cv2
import csv
from imgaug import augmenters as iaa
import warnings
import numpy as np
import random
import copy
import os

#file_path = "input/"  # Windows
abs_path = os.path.dirname(os.path.realpath(__file__))
file_path = abs_path + "/input/"
#file_path = "/home/ycyoon/PycharmProject/sealion_count/input/" #Linux

# Model
detection = "detection"
second_det = "second_detection"
classification = "classification"
# Mode
train = "train"
test = "test"
# Sav_name
no_rcrop = "no_rcrop"

sav_name = "NULL"

seq = iaa.Sequential([
    iaa.Flipud(0.5),
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 2.0)),
    iaa.ContrastNormalization((0.5, 2.0)),
    iaa.Multiply((0.5, 1.5))
])


# Class containining deep neural network related materials
class ct():
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    def __init__(self):
        self.accum_crop = 0
        self.accum_process = 0
        self.s = tf.placeholder(dtype=tf.float32, shape=[None, 16, 16, 3])
        self.m = tf.placeholder(dtype=tf.float32, shape=[None, 16, 16, 3])
        self.b = tf.placeholder(dtype=tf.float32, shape=[None, 16, 16, 3])
        self.cls_img = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
        self.det_gt = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        self.cls_gt = tf.placeholder(dtype=tf.float32, shape=[None, 4])

        self.det_out, self.det_soft = cnn.deepConcatNet(self.s, self.m, self.b, 2, self.keep_prob, self.is_training)
        self.cls_out, self.cls_soft = cnn.deepClassNet(self.cls_img, 4, self.keep_prob, self.is_training)

        self.det_train = cnn.trainStep(self.det_out, self.det_gt)
        self.det_acc = cnn.accuracy(self.det_out, self.det_gt)
        self.cls_train = cnn.trainStep(self.cls_out, self.cls_gt)
        self.cls_acc = cnn.accuracy(self.cls_out, self.cls_gt)

        self.img = tf.placeholder(dtype=tf.float32, shape=[32, 32, 3])

        self.init = tf.global_variables_initializer();
        self.saver = tf.train.Saver()
        self.restored = False

def cropBatch(batch, prev_sz, object_sz, rand):
    if rand == True:
        x_rand = random.randint(-6, 6)
        y_rand = random.randint(-6, 6)
    else:
        x_rand = 0
        y_rand = 0
    interval = int(object_sz/2)
    x_center = int(prev_sz/2) + x_rand
    y_center = int(prev_sz/2) + y_rand
    temp_img = batch[0][x_center-interval:x_center+interval, y_center-interval:y_center+interval]
    temp_batch = np.array([temp_img])
    for i in range(1, len(batch)):
        if rand == True:
            x_rand = random.randint(-6, 6)
            y_rand = random.randint(-6, 6)
        else:
            x_rand = 0
            y_rand = 0
        interval = int(object_sz / 2)
        x_center = int(prev_sz / 2) + x_rand
        y_center = int(prev_sz / 2) + y_rand
        temp_batch = np.append(temp_batch, [batch[i][x_center-interval:x_center+interval, y_center-interval:y_center+interval]], axis = 0)
    return temp_batch

def resizeBatch(batch, size):
    temp_img = transform.resize(batch[0], (size, size), mode = 'reflect')
    temp_batch = np.array([temp_img])
    for i in range(1, len(batch)):
        temp_batch = np.append(temp_batch, [transform.resize(batch[i], (size, size), mode = 'reflect')], axis = 0)
    return temp_batch

# Training
def initTrain(ct, model, mode, local_iter, global_iter):
    if model == classification:
        train_model = ct.cls_train
        gt_model = ct.cls_gt
    else:
        train_model = ct.det_train
        gt_model = ct.det_gt
    ct.restored = True
    whole_batch = batch.list_generator(mode, model)

    for g in range(global_iter):
        print("Iteration : ", g)
        copy_batch = copy.deepcopy(whole_batch)
        print(len(copy_batch[0]))
        for i in range(local_iter):
            sample, label, copy_batch = batch.batch_generator(copy_batch, model, mode)
            #Data augmentation
            sample = seq.augment_images(sample)

            if i%100 == 0:
                print("< ", str(i),"th >")
                # Display accuracy
                if model == classification:
                    print("Accuracy : ", eval(ct, 10, model, "classification"))
                else:
                    tt = eval(ct, 1000, model, "detection_sealion")
                    ff = eval(ct, 2000, model, "detection_background")
                    print("true-true : ", tt)
                    print("false-false : ", ff)
            if model == detection:
                # Expands a patch into three different view patches
                sample = cropBatch(sample, 80, 64, False)
                s_batch = cropBatch(sample, 64, 16, False)
                m_batch = cropBatch(sample, 64, 32, False)
                m_batch = resizeBatch(m_batch, 16)
                b_batch = resizeBatch(sample, 16)
                sess.run(train_model, feed_dict={ct.s:s_batch, ct.m:m_batch, ct.b:b_batch, gt_model:label, ct.keep_prob:0.5, ct.is_training: True})
            elif model == classification:
                sample = cropBatch(sample, 80, 64, True)
                cls_batch = resizeBatch(sample, 32)
                sess.run(train_model, feed_dict={ct.cls_img:cls_batch, gt_model:label, ct.keep_prob:0.5, ct.is_training: True})

        save_path = ct.saver.save(sess, file_path + sav_name + ".ckpt")
        print("Model saved in file: ", save_path)
        print("< Result >")
        if model == classification:
            print("Accuracy : ", eval(ct, 50, model, "classification"))
        else:
            print("true-true : ", eval(ct, 2000, model, "detection_sealion"))
            print("false-false : ", eval(ct, 10000, model, "detection_background"))
    return True

# Restore saved model and re-train
def reTrain(ct, model, mode, local_iter, global_iter):
    if ct.restored == False:
        ct.saver.restore(sess, file_path + sav_name + ".ckpt")
        ct.restored = True
    initTrain(ct, model, mode, local_iter, global_iter)
    print("Retrain done")
    return True

# Evaluation
def eval(ct, iter, sav_model, eval_model):
    if ct.restored == False:
        ct.saver.restore(sess, file_path + sav_name + ".ckpt")
        ct.restored = True
    if sav_model == classification:
        acc_model = ct.cls_acc
        gt_model = ct.cls_gt
    else:
        acc_model = ct.det_acc
        gt_model = ct.det_gt

    whole_batch = batch.list_generator("evaluation", eval_model)
    accum = 0
    for i in range(iter):
        sample, label, whole_batch = batch.batch_generator(whole_batch, eval_model, "evaluation")
        sample = cropBatch(sample, 80, 64, True)
        if sav_model == detection:
            s_batch = cropBatch(sample, 64, 16, True)
            m_batch = cropBatch(sample, 64, 32, True)
            m_batch = resizeBatch(m_batch, 16)
            b_batch = resizeBatch(sample, 16)
            accum = accum + acc_model.eval(feed_dict={ct.s: s_batch, ct.m: m_batch, ct.b: b_batch, gt_model: label,
                                                      ct.keep_prob: 1.0, ct.is_training: False})
        elif sav_model == classification:
            cls_batch = resizeBatch(sample, 32)
            accum = accum + acc_model.eval(feed_dict={ct.cls_img: cls_batch, gt_model: label,
                                                      ct.keep_prob: 1.0, ct.is_training: False})
    return (accum / iter)

#Patch inference
def infer(ct, det_cls, batch):
    if det_cls == "detection":
        prob = ct.det_soft.eval(feed_dict = {ct.s:batch[0], ct.m:batch[1], ct.b:batch[2], ct.keep_prob:1.0, ct.is_training: False})
    elif det_cls == "classification":
        prob = ct.cls_soft.eval(feed_dict={ct.cls_img: batch, ct.keep_prob: 1.0, ct.is_training: False})
    return prob

#Apply detector on whole-size picture
def detectOnImg(ct, operation, st=None, end=None):
    ct.saver.restore(sess, file_path + sav_name + ".ckpt")
    #final_path = "G:/Kaggle dataset/steller sealion/Kaggle-NOAA-SeaLions/Test/"
    ft_count = 0
    tot_ft = 0
    if operation == "ft_generator":
        start_num = 0
        end_num = 750
    elif operation == "test_set":
        if st == None or end == None:
            print("Error : You should give start, end number of image")
            return False
        elif st >= end:
            print("Error : You should give proper start, end number of image")
            return False
        start_num = st
        end_num = end
    elif operation == "demo":
        start_num = 1
        end_num = 10
    train_mis = extract.read_mismatch("full")
    while True:
        if int(train_mis[0]) < start_num & len(train_mis) > 0:
            train_mis.remove(train_mis[0])
        else:
            break

    tot_img_num = end_num-start_num
    for img_num in range(start_num, end_num):
        print("#",str(img_num))
        sealion_num = getSealionNum(img_num)
        gt_total_num = 0
        for i in range(len(sealion_num)):
            gt_total_num += sealion_num[i]
        if len(train_mis) > 0:
            if train_mis[0] == str(img_num):
                train_mis.remove(train_mis[0])
                tot_img_num -= 1
                continue
        ft_arr = []
        if operation == "demo":
            img_path = file_path + "Sample/" + str(img_num) + ".jpg"
        else:
            img_path = file_path + "Train/" + str(img_num) + ".jpg"
        img_crop = io.imread(img_path)

        gt_coord = getCoord(img_num)

        if operation == "ft_generator":
            img_dot = io.imread(file_path + "TrainDotted/" + str(img_num) + ".jpg")
            p_coord, prob_label = sliding_window.get_coord(ct, img_crop, img_dot, "ft_generate", 0.8)
            # Find false positive error patches
            for p_n in range(len(p_coord)):
                indicator = False
                for g_n in range(len(gt_coord)):
                    # Distance btw gt and predicted coordinate
                    d = (int(p_coord[p_n][0])-int(gt_coord[g_n][0]))**2 + (int(p_coord[p_n][1])-int(gt_coord[g_n][1]))**2
                    # Distance threshold
                    if d < 576:
                        indicator = True
                        break
                # If p_n is not close to any gt coordinate, append p_n to the false_positive list
                if (indicator == False) and (p_coord[p_n] not in ft_arr) :
                    ft_arr.append(p_coord[p_n])
            tot_ft += len(ft_arr)
            ft_count = savNewImg(ft_arr, ft_count, img_crop)
        else:
            p_coord, prob_label = sliding_window.get_coord(ct, image=img_crop, mode="test", limit=0.95)
            print("Crop_time", ct.accum_crop, "Process_time", ct.accum_process)
            for iter in range(1):
                for i in range(len(p_coord)):
                    x = p_coord[i][1]
                    y = p_coord[i][0]
                    #red = 255 - (1000*(1- prob_label[i]))
                    #if red < 0:
                        #red = 0
                    #cv2.rectangle(cp_img, ( x -2, y - 2),(x + 2, y + 2), (255, 0, 0), 4)
                #io.imshow(cp_img)
                #io.show()
            classifier(ct, p_coord, img_crop)
    return True

#Age classifier
def classifier(ct, det_coord, img_crop):
    if len(det_coord) == 0:
        print("No sealion in the image")
        return
    cp = np.copy(img_crop)
    batch = []
    iter = 0
    for det_num in range(len(det_coord)):
        y_center = det_coord[det_num][0]
        x_center = det_coord[det_num][1]
        temp_img = cp[y_center-32:y_center+32, x_center-32:x_center+32]
        temp_img = transform.resize(temp_img, (32, 32), mode='reflect')
        if iter == 0:
            batch = np.array([temp_img])
        else:
            batch = np.append(batch, [temp_img], axis = 0)
        iter += 1
    label = infer(ct, "classification", batch)
    predict_num = [0, 0, 0, 0, 0]

    for l_num in range(len(label)):
        x = det_coord[l_num][1]
        y = det_coord[l_num][0]
        r = 0
        g = 0
        b = 0
        #print(label[l_num])
        max_idx = np.argmax(label[l_num])
        # Male adult
        if max_idx == 0:
            r = 200
            g = 0
            b = 0
            predict_num[0] += 1
        # Subadult male
        elif max_idx == 1:
            r = 200
            g = 0
            b = 200
            predict_num[1] += 1
        # Female adult
        elif max_idx == 2:
            r = 120
            g = 50
            b = 30
            predict_num[2] += 1
        # Juvenile
        elif max_idx == 3:
            r = 0
            g = 0
            b = 200
            predict_num[3] += 1
        cv2.rectangle(cp, (x - 2, y - 2), (x + 2, y + 2), (r, g, b), 4)
    io.imshow(cp)
    io.show()
    return True

def savNewImg(ft_arr, ft_count, image):
    ft_path = file_path + "train_set/ft_img/"
    extract.folder_check(ft_path)
    random.shuffle(ft_arr)
    for i in range(len(ft_arr)):
        if i >= 200:
            break
        y_center = int(ft_arr[i][0])
        x_center = int(ft_arr[i][1])
        if y_center < 40 or y_center > len(image) - 40 or x_center < 40 or x_center > len(image[0]) - 40:
            continue
        crop = image[y_center - 40:y_center + 40, x_center - 40:x_center + 40]
        ft_count += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(ft_path + str(ft_count) + ".jpg", crop)
    return ft_count

def getCoord(img_num):
    f = open(file_path + "coords.csv")
    reader = csv.reader(f)
    arr = []
    iter = 0
    for row in reader:
        if iter == 0:
            iter += 1
            continue
        if len(row) != 0:
            if int(row[0]) == img_num and int(row[1]) <= 3:
                arr.append([row[2],row[3]])
            elif int(row[0]) > img_num:
                break
    return arr

def getSealionNum(img_num):
    f = open(file_path + "train.csv")
    reader = csv.reader(f)
    row = next(reader)
    sealion_num = [0, 0, 0, 0, 0]
    for row in reader:
        if len(row) != 0:
            if int(row[0]) == img_num:
                for col in range(1, 6):
                    sealion_num[col-1] += int(row[col])
            elif int(row[0]) > img_num:
                break
    return sealion_num

if __name__ == "__main__":
    # model : detection, second_det, classification, non_flip
    model = detection
    # mode : train, test
    mode = train
    img_sz = 16

    sav_name = detection + str(img_sz)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    ct = ct()

    with sess:
        sess.run(ct.init)

        detectOnImg(ct, "demo")
        #detectOnImg(ct, "test_set", 750, 948)

        #initTrain(ct, model, train, 1600, 100)
        #detectOnImg(ct, "ft_generator")
        #reTrain(ct, classification, train, 300, 150)
        #reTrain(ct, model, train, 1600, 1)
