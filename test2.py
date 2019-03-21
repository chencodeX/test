# -*- coding: utf-8 -*-
# Copyright (c) 2016 - zihao.chen <zihao.chen@moji.com> 

"""
Author: zihao.chen
Create Date: 2019/2/28
Modify Date: 2019/2/28
descirption:
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

all_locations = []


def video2img():
    cap = cv2.VideoCapture('move3.mp4')
    index = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('imgs3/gray_%d.png' % index, gray)
            print index
        else:
            break
        # print frame.shape

        # cv2.imwrite('imgs/%d.png'%index,frame)
        index += 1

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #
        # cv2.imshow('frame',gray)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()

#第一步 先把视频解析为单独的灰度帧
# video2img()


def hist_img(image_all):
    image = image_all
    max_pix = np.max(image)
    min_pix = np.min(image)
    a = float((255 - 0) / max_pix - min_pix)
    b = 0 - a * min_pix
    output_img = a * image + b
    output_img = output_img.astype(np.uint8)
    image_all = output_img
    return image_all


def hist_img1(image_gray):
    clache = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    dst = clache.apply(image_gray)

    return dst


# 圆检测
def circles_image(image):
    global all_locations
    # dst = cv2.pyrMeanShiftFiltering(image, 10, 15)
    # image = image.astype(np.float)
    image = hist_img1(image)
    # image[300:500,400:600,:] = np.clip((1.6 * image[300:500,400:600,:]), 0, 255)

    # image = np.uint8(image)
    # plt.imshow(image)
    # plt.show()
    # cimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plt.imshow(cimage)
    # plt.show()
    # print cimage.mean()
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=60, maxRadius=70)
    print circles[0][0]
    all_locations.append(circles[0][0])
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv2.circle(image, (i[0], i[1]), 2, (255, 0, 255), 2)

    return image


# img = cv2.imread('imgs1/2.png')
# test_img1 = cv2.imread('imgs1/gray_0.png',0).astype(np.float)
# test_img2 = cv2.imread('imgs1/gray_1.png',0).astype(np.float)
# cv2.pyrMeanShiftFiltering()

# results = cv2.cornerHarris(test_img,2,3,0.04)

# print results
# plt.subplot(121), plt.imshow(imgs[i]), plt.title(titles[i])
# cic_img = circles_image(img)
# result_img = test_img2-test_img1
# plt.imshow(cic_img)
# plt.show()
def calcula_distance(local1, local2):
    length = math.sqrt(math.pow(math.fabs(local1[0] - local2[0]), 2) + math.pow(math.fabs(local1[1] - local2[1]), 2))
    return length

#第二步找圆心
for i in range(0, 310):
    img_path = 'imgs3/gray_%d.png' % i
    cimg = cv2.imread(img_path, 0)
    c_cimg = circles_image(cimg)
    # cv2.imwrite('imgs3/circ_%d.png' % i, c_cimg)

#第三步 算距离
all_length = []
for index in range(1, len(all_locations)):
    old_local = all_locations[index - 1]
    now_local = all_locations[index]
    all_length.append(calcula_distance(old_local, now_local))

#展示结果
print all_length
t_data = np.array(all_locations)
print t_data.mean(axis=0)
x = range(len(all_length))
p_, = plt.plot(x, all_length)
l1 = plt.legend([p_],[u"data_delay"], loc='speed', markerscale=1)
plt.grid(linestyle='--', linewidth=1, alpha=0.3)
plt.savefig("speed1.png")
plt.show()
