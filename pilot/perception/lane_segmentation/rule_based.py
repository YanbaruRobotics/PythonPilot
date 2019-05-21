#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File        : rule_based.py
# Environment : Python 3.5.2
#               OpenCV 3.4.5
"""
Rule Based Lane Segmentation
"""

import cv2
import numpy as np
import os
import time

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from perception_common import handling_image


class RuleBased(object):

    def __init__(self):
        self.__width = 512
        self.__height = 256

    def __apply_canny(self, src, ksize=7, sigma=1.2, low_th=10, high_th=70):
        """Apply canny edge detection.

        Args:
            src (int): Input image BGR.
                       numpy.ndarray, (720, 1280, 3), 0~255

        Returns:
            dst (int): Output image.
                       numpy.ndarray, (720, 1280), 0~1

        """
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        blur_gray = cv2.GaussianBlur(gray,(ksize, ksize), sigma)
        dst = cv2.Canny(blur_gray, low_th, high_th) // 255

        return dst

    def __apply_multi_threshold(self, src):
        """Apply multi thresholding using LAB, HLS and HSV.

        Args:
            src (int): Input image BGR.
                       numpy.ndarray, (720, 1280, 3), 0~255

        Returns:
            dst (int): Output image.
                       numpy.ndarray, (720, 1280), 0~1

        """
        settings = []
        settings.append({'cspace': 'LAB', 'channel': 2, 'clipLimit': 2.0, 'threshold': 190})
        settings.append({'cspace': 'HLS', 'channel': 1, 'clipLimit': 1.0, 'threshold': 200})
        settings.append({'cspace': 'HSV', 'channel': 2, 'clipLimit': 3.0, 'threshold': 230})

        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        dst = np.zeros_like(gray)
        for s in settings:
            color_t = getattr(cv2, 'COLOR_RGB2{}'.format(s['cspace']))
            gray = cv2.cvtColor(src, color_t)[:,:,s['channel']]
            
            clahe = cv2.createCLAHE(s['clipLimit'], tileGridSize=(8,8))
            norm_img = clahe.apply(gray)
            
            binary = np.zeros_like(norm_img)
            binary[(norm_img >= s['threshold']) & (norm_img <= 255)] = 1
            dst[(dst == 1) | (binary == 1)] = 1

        return dst

    def __mask_image(self, src, top_rate=0.486, bottom_rate=0.833):
        """Mask top and bottom area of input image.

        Args:
            src (int): Input image.
                       numpy.ndarray, (720, 1280), 0~1

        Returns:
            dst (int): Output image.
                       numpy.ndarray, (720, 1280), 0~1

        """
        top = int(src.shape[0] * top_rate)
        bottom = int(src.shape[0] * bottom_rate)

        dst = src.copy()*0
        dst[top:bottom,:] = src[top:bottom,:]
        return dst

    def binarize(self, src):
        """Image binarization.

        Args:
            src (int): Input image BGR.
                       numpy.ndarray, (256, 512, 3), 0~255

        Returns:
            dst (int): Output image.
                       numpy.ndarray, (256, 512), 0~1

        """
        # resize
        if src.shape[0] != self.__height or src.shape[1] != self.__width:
            src = cv2.resize(src, (self.__width, self.__height), interpolation=cv2.INTER_LINEAR)

        # image thresholding
        image_binary_canny = self.__apply_canny(src)
        image_binary_mthreshold = self.__apply_multi_threshold(src)
        image_binary_all = cv2.bitwise_or(image_binary_canny, image_binary_mthreshold)

        # mask top and bottom
        dst = self.__mask_image(image_binary_all)

        return dst


if __name__ == "__main__":
    """Rule based segmantation test.

    """

    # parameter
    image_dir = "../../../data/input_images/test_image/"
    processing_loop_n = 30
    show_image =  {
      'binarized': True,
      'overlayed': True
    }

    # instance
    rb = RuleBased()

    # load input image
    image_name = os.listdir(image_dir)
    image_path = (image_dir + image_name[-1])
    input_image = cv2.imread(image_path)

    # segmentation
    cost_time_mean = []
    for i in range(processing_loop_n):
        start_time = time.time()
        image_binarized = rb.binarize(input_image)
        cost_time = time.time() - start_time
        cost_time_mean.append(cost_time)
    print('Processing time per image [sec]:', format(np.mean(cost_time_mean), '.3f'))
    cost_time_mean.clear()

    # overlay
    image_overlayed = handling_image.binary_mask_overlay(input_image, image_binarized)

    # show image
    while cv2.waitKey(0) < 0:
        if not show_image['binarized'] and not show_image['overlayed']:
            break
        else:
            if show_image['binarized']:
                cv2.imshow('binarized', image_binarized*255)
            if show_image['overlayed']:
                cv2.imshow('overlayed', image_overlayed)
    cv2.destroyAllWindows()
