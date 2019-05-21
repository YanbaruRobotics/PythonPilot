#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File        : handling_image.py
# Environment : Python 3.5.2
#               OpenCV 3.4.5
"""
"""


import cv2
import numpy as np


def image_property(arg):
    return str(type(arg)) + '\t' + str(arg.dtype) + '\t' + str(arg.shape) + '\t' + str(arg.min()) + '~' + str(arg.max())


def mask_overlay(src, mask):
    """Image overlaying.

    Args:
        src (int): Input image BGR.
            numpy.ndarray, (720, 1280, 3), 0~255
        mask (int): Input image BGR.
            numpy.ndarray, (720, 1280, 3), 0~255

    Returns:
        dst (int): Output image BGR.
                   numpy.ndarray, (720, 1280, 3), 0~255

    """
    # binarize
    gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    ret, mask_bynary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_bynary = mask_bynary // 255

    # resize
    if src.shape[0] != mask.shape[0] or src.shape[1] != mask.shape[1]:
        mask = cv2.resize(mask, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask_bynary = cv2.resize(mask_bynary, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LINEAR)

    # prepare white image
    mask_white = np.dstack((mask_bynary*255, mask_bynary*255, mask_bynary*255))

    # combine src and thresholded image
    not_mask = cv2.bitwise_not(mask_white)
    src_masked = cv2.bitwise_and(src, not_mask)
    dst = cv2.bitwise_or(src_masked, mask)

    return dst


def binary_mask_overlay(src, binary_mask):
    """Image overlaying.

    Args:
        src (int): Input image BGR.
            numpy.ndarray, (720, 1280, 3), 0~255
        binary_mask (int): Input image.
            numpy.ndarray, (720, 1280), 0~1

    Returns:
        dst (int): Output image BGR.
                   numpy.ndarray, (720, 1280, 3), 0~255

    """
    # resize
    if src.shape[0] != binary_mask.shape[0] or src.shape[1] != binary_mask.shape[1]:
        binary_mask = cv2.resize(binary_mask, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LINEAR)

    # prepare white and red image
    image_all_mid_white = np.dstack((binary_mask*255, binary_mask*255, binary_mask*255))
    image_all_mid_red = np.dstack((binary_mask*0, binary_mask*0, binary_mask*255))

    # combine src and thresholded image
    not_mask = cv2.bitwise_not(image_all_mid_white)
    src_masked = cv2.bitwise_and(src, not_mask)
    dst = cv2.bitwise_or(src_masked, image_all_mid_red)

    return dst


def over_lighten(src, mask):
    # resize
    if src.shape[0] != mask.shape[0] or src.shape[1] != mask.shape[1]:
        mask = cv2.resize(mask, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LINEAR)

    is_BG_lighter = src > mask
    result = np.zeros(src.shape, dtype='uint8')
    result[is_BG_lighter] = src[is_BG_lighter]
    result[~is_BG_lighter] = mask[~is_BG_lighter]
    return result


def from1280to512(value, axis):
    if axis == 0:
        value = value / 1280 * 512
    elif axis == 1:
        value = value / 720 * 256
    return value


def from512to1280(value, axis):
    if axis == 0:
        value = value / 512 * 1280
    elif axis == 1:
        value = value / 256 * 720
    return value


def setPerspectiveTransform(cfg, width=512, height=256):
    top_left = [0] * 2
    top_right = [0] * 2
    bottom_left = [0] * 2
    bottom_right = [0] * 2

    # 1280, 720 -> 512, 256
    for i in range(2):
        top_left[i] = from1280to512(cfg['top_left'][i], i)
        top_right[i] = from1280to512(cfg['top_right'][i], i)
        bottom_left[i] = from1280to512(cfg['bottom_left'][i], i)
        bottom_right[i] = from1280to512(cfg['bottom_right'][i], i)

    top_height = (top_left[1] + top_right[1]) / 2.0 / height
    bottom_height = (bottom_left[1] + bottom_right[1]) / 2.0 / height
    top_width = (top_right[0] - top_left[0]) / width
    bottom_width = (bottom_right[0] - bottom_left[0]) / width

    inputQuad = [[width / 2.0 * (1.0 - top_width) + cfg['head_calib'], height * top_height],    # top left
                 [width / 2.0 * (1.0 - bottom_width), height * bottom_height],                  # bottom left
                 [width / 2.0 * (1.0 + bottom_width), height * bottom_height],                  # bottom right
                 [width / 2.0 * (1.0 + top_width) + cfg['head_calib'], height * top_height]]    # top right

    outputQuad = [[width * 0.45, 0.0],
                  [width * 0.45, height],
                  [width * (1.0 - 0.45), height],
                  [width * (1.0 - 0.45), 0.0]]

    src = np.float32(inputQuad)
    dst = np.float32(outputQuad)

    trans_mat33 = cv2.getPerspectiveTransform(src, dst)
    trans_mat33_r = cv2.getPerspectiveTransform(dst, src)

    return trans_mat33, trans_mat33_r


def transformImage(image, M, flags=cv2.INTER_NEAREST):
    img_size = (image.shape[1], image.shape[0])
    return cv2.warpPerspective(image, M, img_size, flags=flags)


def set_array_cordinate_cluster_to_camera(input_array, array_width=128, array_height=25):
    camera_cordinated_input_array = np.zeros(input_array.shape, dtype = 'float')

    if input_array.shape[1] != 2:
        return False

    camera_cordinated_input_array[:,0] = input_array[:,0] / array_width * 512
    camera_cordinated_input_array[:,1] = input_array[:,1] / array_height * 256

    return camera_cordinated_input_array


def set_array_cordinate_camera_to_vehicle(input_array):
    """
    x: 60 pixel = 3.7 m
    y: 256 pixel = 100.0 m
    """
    vehicle_cordinated_input_array = np.zeros(input_array.shape, dtype = 'float')
    input_array = np.array(input_array, dtype = 'float')

    if input_array.shape[1] != 2:
        return False

    vehicle_cordinated_input_array[:,0] = (input_array[:,0] - 256.0) / 60.0 * 3.7
    vehicle_cordinated_input_array[:,1] = (256.0 - input_array[:,1]) / 256.0 * 100.0

    return vehicle_cordinated_input_array


def set_array_cordinate_vehicle_to_camera(input_array):
    """
    x: 60 pixel = 3.7 m
    y: 256 pixel = 100.0 m
    """
    camera_cordinated_input_array = np.zeros(input_array.shape, dtype = 'float')

    if input_array.shape[1] != 2:
        return False

    camera_cordinated_input_array[:,0] = input_array[:,0] / 3.7 * 60.0 + 256.0
    camera_cordinated_input_array[:,1] = -input_array[:,1] / 100.0 * 256.0 + 256.0

    camera_cordinated_input_array = np.array(camera_cordinated_input_array, dtype = 'int64')

    return camera_cordinated_input_array
