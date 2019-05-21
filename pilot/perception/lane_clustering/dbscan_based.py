#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File        : dbscan_based.py
# Environment : Python 3.5.2
#               OpenCV 3.4.5
"""
Lane Clustering using DBSCAN
"""

import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from sklearn.cluster import DBSCAN

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lane_segmentation import RuleBased
from lane_segmentation import LaneNet
from perception_common import handling_image


class DbscanBased(object):

    def __init__(self, eps=4, width=256, height=64):
        self.dbscan = DBSCAN(eps=eps)
        self.__compress_width = width
        self.__compress_height = height
        self.dbscan_input_image = np.zeros((height, width, 1), np.uint8)

    def clusterize(self, src):
        """Image binarization.

        Args:
            src (int): Input bynary image.
                       numpy.ndarray, (256, 512), 0~1

        """
        # top filter
        cv2.rectangle(src, (0, 0), (512, 170), (0, 0, 0), thickness=-1)

        # resize
        self.dbscan_input_image = cv2.resize(src, (self.__compress_width, self.__compress_height), interpolation=cv2.INTER_LINEAR)

        # convert image to array
        dbscan_input_image = self.dbscan_input_image
        dbscan_input_image_nonzero = dbscan_input_image.nonzero()
        dbscan_input_image_nonzero_array = np.array([dbscan_input_image_nonzero[1], dbscan_input_image_nonzero[0]]) 
        dbscan_input_array = dbscan_input_image_nonzero_array.transpose()

        # clustering
        if len(dbscan_input_array) > 0:
            dbscan_label = self.dbscan.fit_predict(dbscan_input_array)
            dbscan_label_n = np.max(dbscan_label) + 1
        else:
            dbscan_label = 0
            dbscan_label_n = 0

        return dbscan_input_array, dbscan_label, dbscan_label_n

    def draw(self, dbscan_input_array, dbscan_label, dbscan_label_n):

        # convert array to image
        frame_draw = np.zeros((self.__compress_height, self.__compress_width), np.uint8)
        frame_draw = cv2.cvtColor(frame_draw, cv2.COLOR_GRAY2RGB)
        for i in range(dbscan_input_array.shape[0]):
            if not dbscan_label[i] == -1:
                color_th = dbscan_label[i] / dbscan_label_n
                c_r = int(cm.hsv(color_th)[0]*255)
                c_g = int(cm.hsv(color_th)[1]*255)
                c_b = int(cm.hsv(color_th)[2]*255)
                frame_draw = cv2.circle(frame_draw, \
                                        (int(dbscan_input_array[i][0]), \
                                         int(dbscan_input_array[i][1])), \
                                        1, (c_r, c_g, c_b), 1)

        return frame_draw


if __name__ == "__main__":
    """DBSCAN based clustering test.

    """

    # parameter
    image_dir = "../../../data/input_images/test_image/"
    processing_loop_n = 20
    show_image =  {
      'binarized': True,
      'clusterized': True,
      'overlayed': True
    }

    # instance
    # segmentator = RuleBased()
    segmentator = LaneNet()
    cluster = DbscanBased()

    # prepare input image
    image_name = os.listdir(image_dir)
    image_path = (image_dir + image_name[3])
    input_image = cv2.imread(image_path)

    # segmentation
    image_binarized = segmentator.binarize(input_image)

    # transfrom
    trans_mat33_, trans_mat33_r_ = handling_image.setPerspectiveTransform()
    image_binarized_transformed = handling_image.transformImage(image_binarized, trans_mat33_)

    # clustering
    print('Processing count:', processing_loop_n)
    cost_time_mean = []
    for i in range(processing_loop_n):
        start_time = time.time()
        dbscan_input_array, dbscan_label, dbscan_label_n = cluster.clusterize(image_binarized_transformed)
        cost_time = time.time() - start_time
        cost_time_mean.append(cost_time)
    print('dbscan_input_array_size', dbscan_input_array.size)
    print('dbscan_label_n', dbscan_label_n)
    print('Processing time per image [sec]:', format(np.mean(cost_time_mean), '.3f'))
    cost_time_mean.clear()

    # draw
    image_clusterized = cluster.draw(dbscan_input_array, dbscan_label, dbscan_label_n)

    # inverse transform
    image_clusterized_512 = cv2.resize(image_clusterized, (512, 256), interpolation=cv2.INTER_LINEAR)
    image_clusterized_transformed = handling_image.transformImage(image_clusterized_512, trans_mat33_r_)

    # overlay
    image_overlayed = handling_image.mask_overlay(input_image, image_clusterized_transformed)

    # show image
    while cv2.waitKey(0) < 0:
        if not show_image['binarized'] and not show_image['clusterized'] and not show_image['overlayed']:
            break
        else:
            if show_image['binarized']:
                cv2.imshow('binarized', image_binarized*255)
            if show_image['clusterized']:
                cv2.imshow('clusterized', image_clusterized)
            if show_image['overlayed']:
                cv2.imshow('overlayed', image_overlayed)
    cv2.destroyAllWindows()
