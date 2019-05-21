#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File        : leasts_based.py
# Environment : Python 3.5.2
#               OpenCV 3.4.5
"""
Lane Fitting using Least Square
"""

import cv2
import collections
import time
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from sklearn import linear_model
import pandas as pd

import sys, os
sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))

from lane_segmentation import RuleBased
from lane_segmentation import LaneNet
from lane_clustering import DbscanBased
from perception_common import handling_image as himg


class LeastsBased(object):

    def __init__(self):
        self.leasts = linear_model.LinearRegression()

        self.__lane_l_abc_lpf = np.array([0.0, 0.0, -3.7/2.0])
        self.__lane_r_abc_lpf = np.array([0.0, 0.0, 3.7/2.0])
        self.__lane_m_abc_lpf = np.array([0.0, 0.0, 0.0])

        self.__lane_width_lpf = 3.7
        self.__lane_z_lpf = 0.1

    def change_cluster_cordinate(self, dbscan_input_array, dbscan_input_array_width, dbscan_input_array_height):
        # transform cordinate cluster to camera
        camera_cordinated_array = himg.set_array_cordinate_cluster_to_camera(dbscan_input_array, \
                                                            array_width=dbscan_input_array_width, \
                                                            array_height=dbscan_input_array_height)
        # transform cordinate camera to vehicle
        vehicle_cordinated_array = himg.set_array_cordinate_camera_to_vehicle(camera_cordinated_array)

        return vehicle_cordinated_array

    def make_lane_list(self, vehicle_cordinated_array, dbscan_label, dbscan_label_n, inner_lane_width=1.5, b_is_zero=False):
        lane_l_list = []
        lane_r_list = []
        lane_dict = {}

        for i in range(dbscan_label_n):
            # group array
            vehicleX = vehicle_cordinated_array[dbscan_label == i][:,0]
            vehicleZ = vehicle_cordinated_array[dbscan_label == i][:,1]
            
            # group basic property
            lane_dict['label'] = i
            lane_dict['points'] = vehicleZ.size
            lane_dict['z_max'] = np.max(vehicleZ)
            lane_dict['z_min'] = np.min(vehicleZ)
            
            # calc center
            lane_dict['center_x'] = np.average(vehicleX)
            lane_dict['center_z'] = np.average(vehicleZ)

            # calc inner array
            target_array = vehicle_cordinated_array[dbscan_label == i]
            target_array_z_list = collections.Counter(target_array[:,1])
            target_array_inner = []
            target_array_x_width_max = 0

            for z in target_array_z_list.most_common():
                target_array_searched = target_array[target_array[:,1] == z[0]]
                target_array_searched_x_min = np.min(target_array_searched[:,0])
                target_array_searched_x_max = np.max(target_array_searched[:,0])
                target_array_searched_x_width = target_array_searched_x_max - target_array_searched_x_min

                # searched inner array
                if lane_dict['center_x'] > 0:
                    target_array_searched_inner = target_array_searched[target_array_searched[:,0] \
                                                                        <= (target_array_searched_x_min + inner_lane_width)]
                else:
                    target_array_searched_inner = target_array_searched[target_array_searched[:,0] \
                                                                        >= (target_array_searched_x_max - inner_lane_width)]
                # add to inner array
                if len(target_array_inner) == 0:
                    target_array_inner = target_array_searched_inner
                else:
                    target_array_inner = np.vstack((target_array_inner, target_array_searched_inner))

                # calc x_width_max
                if target_array_searched_x_width > target_array_x_width_max:
                    target_array_x_width_max = target_array_searched_x_width

            lane_dict['x_width_max'] = target_array_x_width_max
            
            # inner group array
            vehicleX_inner = target_array_inner[:,0]
            vehicleZ_inner = target_array_inner[:,1]

            # fitting prepare
            vehicleZ_1d = vehicleZ_inner.reshape(-1, 1)
            if b_is_zero:
                vehicleZ_2d = np.concatenate([(vehicleZ_1d**2).reshape(-1,1)], axis=1)
            else:
                vehicleZ_2d = np.concatenate([(vehicleZ_1d**2).reshape(-1,1),vehicleZ_1d], axis=1)

            # least square
            try:
                self.leasts.fit(vehicleZ_2d, vehicleX_inner)
                lane_dict['status'] = True
                if b_is_zero:
                    lane_dict['leasts_abc'] = np.array([self.leasts.coef_[0], \
                                                        0.0, \
                                                        self.leasts.predict(np.array([[0]]))[0]])
                else:
                    lane_dict['leasts_abc'] = np.array([self.leasts.coef_[0], \
                                                        self.leasts.coef_[1], \
                                                        self.leasts.predict(np.array([[0, 0]]))[0]])
            except:
                lane_dict['status'] = False
                lane_dict['leasts_abc'] = np.array([0.0, 0.0, 0.0])

            # calc center inner
            lane_dict['center_x_inner'] = np.average(vehicleX_inner)
            lane_dict['center_z_inner'] = np.average(vehicleZ_inner)

            # calc distance z_min
            lane_dict['distance_z_min'] = (lane_dict['center_x'] * lane_dict['center_x'] \
                                                    + lane_dict['z_min'] * lane_dict['z_min']) ** 0.5

            # add to list
            if lane_dict['leasts_abc'][2] < 0.0 and lane_dict['center_x_inner'] < 0.0:
                lane_l_list.append(lane_dict.copy())
            elif lane_dict['leasts_abc'][2] >= 0.0 and lane_dict['center_x_inner'] >= 0.0:
                lane_r_list.append(lane_dict.copy())

        # sort
        if len(lane_l_list) > 1:
            lane_l_list = sorted(lane_l_list, key=lambda x:x['leasts_abc'][2])
        if len(lane_r_list) > 1:
            lane_r_list = sorted(lane_r_list, key=lambda x:x['leasts_abc'][2])

        return lane_l_list, lane_r_list

    def search_nearest_lane(self, lane_l_list, lane_r_list, points_min=5, c_range=2.7):
        lane_l_dict_nearest = {}
        lane_r_dict_nearest = {}

        if len(lane_l_list):
            nearest_distance_z = lane_l_list[-1]['distance_z_min']
            for lane_dict in lane_l_list:
                if abs(lane_dict['leasts_abc'][2]) < c_range \
                    and lane_dict['distance_z_min'] <= nearest_distance_z \
                    and lane_dict['points'] > points_min:
                    nearest_distance_z = lane_dict['distance_z_min']
                    lane_l_dict_nearest = lane_dict

        if len(lane_r_list):
            nearest_distance_z = lane_r_list[0]['distance_z_min']
            for lane_dict in lane_r_list:
                if abs(lane_dict['leasts_abc'][2]) < c_range \
                    and lane_dict['distance_z_min'] <= nearest_distance_z \
                    and lane_dict['points'] > points_min:
                    nearest_distance_z = lane_dict['distance_z_min']
                    lane_r_dict_nearest = lane_dict

        return lane_l_dict_nearest, lane_r_dict_nearest

    def lane_limit(self, lane_dict, l_or_r='none', abc_lim=[0.005, 0.05, 3.7]):
        if len(lane_dict):
            # limit a and b
            for i in range(2):
                lane_dict['leasts_abc'][i] = np.clip(lane_dict['leasts_abc'][i], -abc_lim[i], abc_lim[i])
            # limit c
            if l_or_r == 'l':
                lane_dict['leasts_abc'][2] = np.clip(lane_dict['leasts_abc'][2], -abc_lim[2], 0)
            elif l_or_r == 'r':
                lane_dict['leasts_abc'][2] = np.clip(lane_dict['leasts_abc'][2], 0, abc_lim[2])
        return lane_dict

    def update_width_and_z(self, lane_l_dict, lane_r_dict, lpf_gain=0.08):
        if len(lane_l_dict) and len(lane_r_dict):
            new_width = -np.copy(lane_l_dict['leasts_abc'][2]) + np.copy(lane_r_dict['leasts_abc'][2])
            z_min = min(lane_l_dict['z_max'], lane_r_dict['z_max'])
            z_max = max(lane_l_dict['z_max'], lane_r_dict['z_max'])
            new_z = z_min + (z_max - z_min)
        elif len(lane_l_dict):
            new_width = self.__lane_width_lpf
            new_z = lane_l_dict['z_max']
        elif len(lane_r_dict):
            new_width = self.__lane_width_lpf
            new_z = lane_r_dict['z_max']
        else:
            new_width = 3.7
            new_z = 0.0

        self.__lane_width_lpf = new_width * lpf_gain + self.__lane_width_lpf * (1.0 - lpf_gain)
        self.__lane_z_lpf = new_z * lpf_gain + self.__lane_z_lpf * (1.0 - lpf_gain)
        return

    def lane_lpf(self, lane_l_dict, lane_r_dict, lpf_gain=0.12):
        # lane l
        if len(lane_l_dict):
            lane_l_dict['leasts_abc_lpf'] = np.copy(lane_l_dict['leasts_abc']) * lpf_gain + self.__lane_l_abc_lpf * (1.0 - lpf_gain)
            self.__lane_l_abc_lpf = lane_l_dict['leasts_abc_lpf']
        else:
            self.__lane_l_abc_lpf = np.array([0.0, 0.0, -self.__lane_width_lpf/2.0])

        # lane r
        if len(lane_r_dict):
            lane_r_dict['leasts_abc_lpf'] = np.copy(lane_r_dict['leasts_abc']) * lpf_gain + self.__lane_r_abc_lpf * (1.0 - lpf_gain)
            self.__lane_r_abc_lpf = lane_r_dict['leasts_abc_lpf']
        else:
            self.__lane_r_abc_lpf = np.array([0.0, 0.0, self.__lane_width_lpf/2.0])

        return lane_l_dict, lane_r_dict

    def calc_midle_lane(self, lane_l_dict, lane_r_dict, lpf_gain=0.12, switch_ab=False, switch_c=True):
        lane_dict = {}

        # two lanes
        if len(lane_l_dict) and len(lane_r_dict):
            lane_dict['status'] = 'merged'

            # brend abc all
            lane_dict['leasts_abc_lpf'] = (np.copy(lane_l_dict['leasts_abc_lpf']) + np.copy(lane_r_dict['leasts_abc_lpf'])) / 2.0

            # switch a and b
            if switch_ab:
                if abs(lane_l_dict['leasts_abc_lpf'][0]) < abs(lane_r_dict['leasts_abc_lpf'][0]) \
                    and abs(lane_l_dict['distance_z_min']) < abs(lane_r_dict['distance_z_min']):
                    lane_dict['leasts_abc_lpf'][0] = lane_l_dict['leasts_abc_lpf'][0]
                    lane_dict['leasts_abc_lpf'][1] = lane_l_dict['leasts_abc_lpf'][1]
                elif abs(lane_l_dict['leasts_abc_lpf'][0]) > abs(lane_r_dict['leasts_abc_lpf'][0]) \
                    and abs(lane_l_dict['distance_z_min']) > abs(lane_r_dict['distance_z_min']):
                    lane_dict['leasts_abc_lpf'][0] = lane_r_dict['leasts_abc_lpf'][0]
                    lane_dict['leasts_abc_lpf'][1] = lane_r_dict['leasts_abc_lpf'][1]

            # switch c
            if switch_c:
                ego_width = -lane_l_dict['leasts_abc_lpf'][2] + lane_r_dict['leasts_abc_lpf'][2]
                if ego_width > 3.7:
                    if abs(lane_l_dict['leasts_abc_lpf'][2]) < abs(lane_r_dict['leasts_abc_lpf'][2]):
                        lane_dict['leasts_abc_lpf'][2] = lane_l_dict['leasts_abc_lpf'][2] + self.__lane_width_lpf / 2.0
                    else:
                        lane_dict['leasts_abc_lpf'][2] = lane_r_dict['leasts_abc_lpf'][2] - self.__lane_width_lpf / 2.0

        # one side l
        elif len(lane_l_dict):
            lane_dict['status'] = 'left'
            lane_dict['leasts_abc_lpf'] = np.copy(lane_l_dict['leasts_abc_lpf'])
            lane_dict['leasts_abc_lpf'][2] = lane_l_dict['leasts_abc_lpf'][2] + self.__lane_width_lpf / 2.0

        # one side r
        elif len(lane_r_dict):
            lane_dict['status'] = 'right'
            lane_dict['leasts_abc_lpf'] = np.copy(lane_r_dict['leasts_abc_lpf'])
            lane_dict['leasts_abc_lpf'][2] = lane_r_dict['leasts_abc_lpf'][2] - self.__lane_width_lpf / 2.0

        # none
        else:
            lane_dict['status'] = False
            lane_dict['leasts_abc_lpf'] = np.array([0.0, 0.0, 0.0])

        # lpf
        lane_dict['leasts_abc_lpf'] = np.copy(lane_dict['leasts_abc_lpf']) * lpf_gain + self.__lane_m_abc_lpf * (1.0 - lpf_gain)
        self.__lane_m_abc_lpf = lane_dict['leasts_abc_lpf']

        # store lane width
        lane_dict['lane_width_lpf'] = self.__lane_width_lpf

        # store z
        lane_dict['lane_z_lpf'] = self.__lane_z_lpf

        return lane_dict

    def fitting(self, dbscan_input_array, dbscan_label, dbscan_label_n, \
                dbscan_input_array_width, dbscan_input_array_height):

        # prepate vehicle cordinate
        vehicle_cordinated_array = self.change_cluster_cordinate(dbscan_input_array,\
                                                                 dbscan_input_array_width, \
                                                                 dbscan_input_array_height)

        # make lane candidate
        lane_l_list, lane_r_list = self.make_lane_list(vehicle_cordinated_array, dbscan_label, dbscan_label_n)

        # search nearest lane lr
        lane_l_dict_nearest, lane_r_dict_nearest = self.search_nearest_lane(lane_l_list, lane_r_list)

        # lane lr abc limit
        lane_l_dict_nearest = self.lane_limit(lane_l_dict_nearest, l_or_r='l')
        lane_r_dict_nearest = self.lane_limit(lane_r_dict_nearest, l_or_r='r')

        # lane lr abc lpf
        lane_l_dict_nearest, lane_r_dict_nearest = self.lane_lpf(lane_l_dict_nearest, lane_r_dict_nearest)

        # update lane property with lpf
        self.update_width_and_z(lane_l_dict_nearest, lane_r_dict_nearest)

        # calc midle lane abc
        lane_m_dict = self.calc_midle_lane(lane_l_dict_nearest, lane_r_dict_nearest)

        return vehicle_cordinated_array, lane_l_dict_nearest, lane_r_dict_nearest, lane_m_dict

    def draw_fitting(self, image_binarized_transformed, trans_mat33_r_, lane_l_dict, lane_r_dict, lane_m_dict, steering_a):
        # color setting
        lane_points_color = (51, 243, 169)
        lane_m_list_color = (243, 169, 51)
        lane_lr_list_color = (169, 51, 243)

        # prepaer image_binarized_transformed
        frame_draw = np.dstack((image_binarized_transformed*lane_points_color[0], \
                                image_binarized_transformed*lane_points_color[1], \
                                image_binarized_transformed*lane_points_color[2]))

        # M lane
        if len(lane_m_dict):
            if lane_m_dict['status']:
                a = np.copy(lane_m_dict['leasts_abc_lpf'][0])
                b = np.copy(lane_m_dict['leasts_abc_lpf'][1])
                c = np.copy(lane_m_dict['leasts_abc_lpf'][2])
                # line_z = np.arange(0.0, self.__lane_z_lpf, 1.0)[:, np.newaxis]
                line_z = np.arange(0.0, lane_m_dict['lane_z_lpf'], 1.0)[:, np.newaxis]
                line_l_x = a*line_z**2 + b*line_z + c - 0.9
                line_r_x = a*line_z**2 + b*line_z + c + 0.9

                road_edge_l_array = np.hstack([line_l_x, line_z])
                road_edge_r_array = np.hstack([line_r_x, line_z])
                road_edge_array = np.vstack([road_edge_l_array, np.flipud(road_edge_r_array)])
                camera_cordinated_road_edge_array = himg.set_array_cordinate_vehicle_to_camera(road_edge_array)
                cv2.fillPoly(frame_draw, np.int_([camera_cordinated_road_edge_array]), lane_m_list_color)

        # L lane
        if len(lane_l_dict):
            if lane_l_dict['status']:
                a = np.copy(lane_l_dict['leasts_abc_lpf'][0])
                b = np.copy(lane_l_dict['leasts_abc_lpf'][1])
                c = np.copy(lane_l_dict['leasts_abc_lpf'][2])
                line_z = np.arange(0.0, lane_l_dict['z_max'], 1.0)[:, np.newaxis]
                line_x = a*line_z**2 + b*line_z + c

                line_array_leasts = np.hstack([line_x, line_z])
                camera_cordinated_line_array_leasts = himg.set_array_cordinate_vehicle_to_camera(line_array_leasts)
                cv2.polylines(frame_draw, [camera_cordinated_line_array_leasts], False, lane_lr_list_color, thickness=2)

        # R lanes lines
        if len(lane_r_dict):
            if lane_r_dict['status']:
                a = np.copy(lane_r_dict['leasts_abc_lpf'][0])
                b = np.copy(lane_r_dict['leasts_abc_lpf'][1])
                c = np.copy(lane_r_dict['leasts_abc_lpf'][2])
                line_z = np.arange(0.0, lane_r_dict['z_max'], 1.0)[:, np.newaxis]
                line_x = a*line_z**2 + b*line_z + c

                line_array_leasts = np.hstack([line_x, line_z])
                camera_cordinated_line_array_leasts = himg.set_array_cordinate_vehicle_to_camera(line_array_leasts)
                cv2.polylines(frame_draw, [camera_cordinated_line_array_leasts], False, lane_lr_list_color, thickness=3)

        # draw steering line
        line_z = np.arange(1.0, 200.0, 1.0)[:, np.newaxis]
        line_x = steering_a*line_z**2
        road_edge_array = np.hstack([line_x, line_z])
        camera_cordinated_road_edge_array = himg.set_array_cordinate_vehicle_to_camera(road_edge_array)
        cv2.polylines(frame_draw, np.int_([camera_cordinated_road_edge_array]), False, (0, 255, 255), thickness=2)

        # transform and resize
        frame_draw_transformed = himg.transformImage(frame_draw, trans_mat33_r_, flags=cv2.INTER_LINEAR)

        return frame_draw_transformed


if __name__ == "__main__":
    """Least based fitting test.

    """

    # parameter setting
    # input_viedo_name = '20180910_124456_input_freeway.mp4'
    # input_viedo_name = '20180910_135722_input_freeway.mp4'
    # input_viedo_name = '20180914_015504_input_night.mp4'

    # input_viedo_name = '20190128_221430_input_freeway_night.mp4'
    # input_viedo_name = '20190128_223456_input_city_night.mp4'
    # input_viedo_name = '20190128_224013_input_city_night.mp4'

    # input_viedo_name = 'MAH00071_freeway_morning.MP4'
    input_viedo_name = 'MAH00072_freeway_to_city_morning.MP4'
    # input_viedo_name = 'MAH00075_forest.MP4'
    # input_viedo_name = 'MAH00076_freeway_evening_shadow.MP4'
    # input_viedo_name = 'MAH00077_freeway_evening_sun.MP4'

    input_viedo_dir = '../../../data/input_videos/'
    output_viedo_dir = '../../../data/output_videos/'
    VIDEO_SAVE = False
    SHIFT_SEC = 5
    VIDEO_FPS = 30

    # input video setting
    input_viedo_path = input_viedo_dir + input_viedo_name
    cap = cv2.VideoCapture(input_viedo_path)
    if not cap.isOpened():
        sys.exit()
    cap.set(1, SHIFT_SEC*VIDEO_FPS)

    # output video setting
    if VIDEO_SAVE:
        start_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        output_viedo_path = output_viedo_dir + 'generated_{:s}_used_'.format(str(start_time)) + input_viedo_name
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        out = cv2.VideoWriter(output_viedo_path, fourcc, VIDEO_FPS, (1280, 720))
        if not out.isOpened():
            sys.exit()

    # instance
    # segmentator = RuleBased()
    segmentator = LaneNet()
    cluster = DbscanBased()
    fitter = LeastsBased()

    trans_mat33_, trans_mat33_r_ = himg.setPerspectiveTransform()

    cost_time_mean = []
    while True:
        start_time = time.time()

        ret, input_image = cap.read()
        if not ret:
            break

        # segmentation
        image_binarized = segmentator.binarize(input_image)

        # transfrom
        image_binarized_transformed = himg.transformImage(image_binarized, trans_mat33_)

        # clustering
        dbscan_input_array, dbscan_label, dbscan_label_n = cluster.clusterize(image_binarized_transformed)

        # fitting
        vehicle_cordinated_array, \
        lane_l_dict_nearest, lane_r_dict_nearest, lane_m_dict = fitter.fitting(dbscan_input_array, dbscan_label, dbscan_label_n, \
                                                                               cluster.dbscan_input_image.shape[1], \
                                                                               cluster.dbscan_input_image.shape[0])

        # cost time
        cost_time = time.time() - start_time
        cost_time_mean.append(cost_time)

        # draw
        image_lanes = fitter.draw_fitting(image_binarized_transformed, trans_mat33_r_, \
                                          lane_l_dict_nearest, lane_r_dict_nearest, lane_m_dict, 0.0)

        # overlay
        image_lanes_1280 = cv2.resize(image_lanes, (1280, 720), interpolation=cv2.INTER_LINEAR)
        image_overlayed = himg.over_lighten(input_image, image_lanes_1280)

        # save
        if VIDEO_SAVE:
            out.write(image_overlayed)

        # show
        cv2.imshow('image_overlayed', image_overlayed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # delay
        # time.sleep(0.05)

    print('Processing time per image [sec]:', format(np.mean(cost_time_mean), '.3f'))
    cost_time_mean.clear()

    cap.release()
    if VIDEO_SAVE:
        out.release()
    cv2.destroyAllWindows()
