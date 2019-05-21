#!/usr/bin/python3
# -*- coding: utf-8 -*-

import ctypes
import math
import matplotlib.cm as cm
import multiprocessing
import numpy as np
import queue as Queue
import time
from datetime import datetime
from matplotlib import pyplot as plt

from common.pilot_gloval_variable import MPVariable
from common.camera_img_pipe import start_receiver


class MPBirdsEye():
    def __init__(self, cfg):
        self.__is_running = multiprocessing.Value(ctypes.c_bool,True)
        self.__m = multiprocessing.Process(target=self.__process, args=())
        self.__m.start()
        return

    def end(self):
        """
        End process
        """
        self.__is_running.value = False
        time.sleep(0.1)
        self.__m.join()
        print('Finish MPVisualization')
        return

    def __process(self):
        """
        update process
        """
        q_in = Queue.Queue()
        start_receiver(MPVariable.bv_in_con, q_in, MPVariable.bv_drop_frames)

        try:
            # wait until get camera image
            while MPVariable.lane_m_width_lpf.value == 0.0:
                time.sleep(1)

            # fig setting
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(1,9), dpi=80)

            # cluster points
            color = cm.brg(np.linspace(0,1,10))
            points = [0]*10
            for i in range(10):
                points[i], = plt.plot([], [], '.', color=color[i])

            # steering
            lane_s, = plt.plot([], [], color='Yellow', linewidth=2, label='lane_s')

            # lanes
            lane_m, = plt.plot([], [], color='Cyan', linewidth=2, label='lane_m')
            lane_l, = plt.plot([], [], color='Magenta', linewidth=2, label='lane_l')
            lane_r, = plt.plot([], [], color='Magenta', linewidth=2, label='lane_r')

            # plt.title("birds-eye_view")
            fig.canvas.set_window_title('BirdsEyesView')
            plt.xlabel("x[m]")
            plt.ylabel("z[m]")
            plt.legend(loc='upper right')
            plt.xlim(-6, 6)
            plt.ylim(0, 50)
            plt.gca().yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)

            while self.__is_running.value:
                if q_in.empty():
                    time.sleep(0.001)
                    continue

                q = q_in.get(block=False)
                if q is None:
                    q_in.task_done()
                    break

                # cluster points
                for i in range(min(q['dbscan_label_n'], 10)):
                    points[i].set_data(q['vehicle_cordinated_array'][q['dbscan_label'] == i][:,0], \
                                       q['vehicle_cordinated_array'][q['dbscan_label'] == i][:,1])
                for i in range(q['dbscan_label_n'], 10):
                    points[i].set_data([], [])

                # line steering
                line_z = np.arange(0.0, 50.0, 1.0)[:, np.newaxis]
                line_x_leasts = MPVariable.can_rx_actual_angle.value * 0.0001*line_z**2
                lane_s.set_data(line_x_leasts, line_z)

                # lane midle
                line_z = np.arange(0.0, 50.0, 1.0)[:, np.newaxis]
                line_x_leasts = MPVariable.lane_m_leasts_abc_lpf_a.value*line_z**2 \
                                + MPVariable.lane_m_leasts_abc_lpf_b.value*line_z \
                                + MPVariable.lane_m_leasts_abc_lpf_c.value
                lane_m.set_data(line_x_leasts, line_z)

                # lane left
                if MPVariable.lane_l_leasts_status.value:
                    line_z = np.arange(0.0, MPVariable.lane_l_leasts_z_max.value, 1.0)[:, np.newaxis]
                    line_x_leasts = MPVariable.lane_l_leasts_abc_lpf_a.value*line_z**2 \
                                    + MPVariable.lane_l_leasts_abc_lpf_b.value*line_z \
                                    + MPVariable.lane_l_leasts_abc_lpf_c.value
                    lane_l.set_data(line_x_leasts, line_z)
                else:
                    lane_l.set_data(0, 0)

                # lane right
                if MPVariable.lane_r_leasts_status.value:
                    line_z = np.arange(0.0, MPVariable.lane_r_leasts_z_max.value, 1.0)[:, np.newaxis]
                    line_x_leasts = MPVariable.lane_r_leasts_abc_lpf_a.value*line_z**2 \
                                    + MPVariable.lane_r_leasts_abc_lpf_b.value*line_z \
                                    + MPVariable.lane_r_leasts_abc_lpf_c.value
                    lane_r.set_data(line_x_leasts, line_z)
                else:
                    lane_r.set_data(0, 0)

                plt.pause(.01)

        except KeyboardInterrupt:
            pass
        finally:
            self.__is_running.value = False
            pass
        return
