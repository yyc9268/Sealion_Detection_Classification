# NOAA SEALION COUNTING CHALLENGE COMPETITION CODE 58th/385
# Author : Young-chul Yoon

import numpy as np
from skimage import transform
import cv2
import control
import time

def getPatches(img, i, j, divisor):
    h = len(img)
    w = len(img[0])
    interv_h = round(h / 8)
    interv_w = round(w / 8)

    x = int(i * interv_w)
    y = int(j * interv_h)

    x_len = int(interv_w)
    y_len = int(interv_h)

    if (i != 7 and j != 7):
        x_len += int(32/divisor)
        y_len += int(32/divisor)
    temp_img = img[y:y+y_len, x:x + x_len]
    iter = 0
    batch = []
    for row in range(0, len(temp_img) - int(64/divisor), int(32/divisor)):
        for col in range(0, len(temp_img[0]) - int(64/divisor), int(32/divisor)):
            crop_img = temp_img[row:(row + int(64 / divisor)), col:(col + int(64 / divisor))]
            if iter == 0:
                batch = np.asarray([crop_img])
            else:
                crop_img = np.asarray([crop_img])
                batch = np.concatenate((batch, crop_img))
            iter = iter + 1
    return batch

# Get sealion coords in full image
def get_coord(ct, image, dotted_img = None, mode = "test", limit = 0.95):
    start = time.time()
    half = transform.resize(image,(round(len(image) / 2), round(len(image[0]) / 2)), mode = 'reflect')
    quarter = transform.resize(image, (round(len(image) / 4), round(len(image[0]) / 4)), mode = 'reflect')

    q_img = image
    q_h = len(q_img)
    q_w = len(q_img[0])
    interv_h = round(q_h / 8)
    interv_w = round(q_w / 8)

    full_coords = []
    prob_label = []
    for i in range(8):
        for j in range(8):
            b_batch = getPatches(quarter, i, j , 4)
            m_batch = getPatches(half, i, j , 2)
            m_batch = control.cropBatch(m_batch, 32, 16, False)
            s_batch = getPatches(image, i, j, 1)
            s_batch = control.cropBatch(s_batch, 64, 16, False)

            prob = control.infer(ct, "detection", [s_batch, m_batch, b_batch])

            x = int(i * interv_w)
            y = int(j * interv_h)
            x_len = int(interv_w)
            y_len = int(interv_h)

            label = []
            for prob_len in range(len(prob)):
                label.append(prob[prob_len][1])
            iter = 0

            if (i != 7 and j != 7):
                x_len += 32
                y_len += 32

            for row in range(y, y + y_len - 64, 32):
                for col in range(x, x + x_len - 64, 32):
                    if (iter >= len(label)):
                        break
                    elif label[iter] > limit:
                        y_center = (row+32)
                        x_center = (col+32)
                        if y_center < 32:
                            y_center = 32
                        elif y_center > len(image)-32:
                            y_center = len(image)-32
                        if x_center < 32:
                            x_center = 32
                        elif x_center > len(image[0])-32:
                            x_center = len(image[0])-32
                        if mode == "test":
                            crop_img = image[y_center - 32:y_center + 32, x_center - 32:x_center + 32]
                        else:
                            crop_img = dotted_img[y_center-32:y_center+32, x_center-32:x_center+32]
                        if np.amin(cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)) != 0:
                            full_coords.append([y_center, x_center])
                            prob_label.append(label[iter])
                    iter = iter + 1

    print(time.time() - start)

    return full_coords, prob_label

# Infer to batch of patches in each partition
def inference(image, ct):
    iter = 0
    batch = []
    for row in range(0, len(image) - 64, 32):
        for col in range(0, len(image[0]) - 64, 32):
            crop_img = image[row:(row + 64), col:(col + 64)]
            if iter == 0:
                batch = np.asarray([crop_img])
            else:
                crop_img = np.asarray([crop_img])
                batch = np.concatenate((batch, crop_img))
            iter = iter + 1
    label = control.infer(ct, "detection", batch)
    return label
