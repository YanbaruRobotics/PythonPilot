#!/usr/bin/python3
# -*- coding: utf-8 -*-

import ctypes
import math
import multiprocessing
import numpy as np
import os
import sys
import time
import yaml
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib import animation as animation

sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))

from common.pilot_gloval_variable import MPVariable


class MPChart():
    def __init__(self, cfg):
        self.__is_running = multiprocessing.Value(ctypes.c_bool,True)
        self.__m_angle = multiprocessing.Process(target=self.__process_angle, args=())
        self.__m_czero = multiprocessing.Process(target=self.__process_czero, args=())
        self.__m_angle.start()
        self.__m_czero.start()
        return

    def end(self):
        """
        End process
        """
        self.__is_running.value = False
        time.sleep(0.1)
        self.__m_angle.join()
        self.__m_czero.join()
        print('Finish MPChart')
        return

    def __process_angle(self, x_n=300):
        """
        update process
        """
        try:
            # wait until get camera image
            while MPVariable.can_tx_target_angle.value == 0.0:
                time.sleep(1)

            # fig setting
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(6,3), dpi=80)

            X = []
            targetY = X*0
            actualY = X*0
            target_angle, = plt.plot(X, targetY, color='Cyan', linewidth=2, label='target_angle')
            actual_angle, = plt.plot(X, actualY, color='Yellow', linewidth=2, label='actual_angle')

            ax = fig.add_subplot(111)
            ax.yaxis.tick_right()

            plt.title("angle")
            fig.canvas.set_window_title('angle')
            plt.xlabel("frame[-]")
            plt.ylabel("angle[deg]")
            plt.legend(loc='upper left')
            plt.xlim(0, x_n)
            plt.ylim(-10, 10)

            while self.__is_running.value:
                if len(X) < x_n:
                    X = np.append(X, MPVariable.camera_frame_counter.value)
                    targetY = np.append(targetY, MPVariable.can_tx_target_angle.value)
                    actualY = np.append(actualY, MPVariable.can_rx_actual_angle.value)
                else:
                    # set new data
                    X[0] = MPVariable.camera_frame_counter.value
                    targetY[0] = MPVariable.can_tx_target_angle.value
                    actualY[0] = MPVariable.can_rx_actual_angle.value
                    # shift
                    X = np.roll(X, -1)
                    targetY = np.roll(targetY, -1)
                    actualY = np.roll(actualY, -1)

                # update axis
                target_angle.set_data(X, targetY)
                actual_angle.set_data(X, actualY)
                plt.xlim(max(X)-x_n, max(X))

                # draw
                plt.pause(.1)

        except KeyboardInterrupt:
            pass
        finally:
            self.__is_running.value = False
            pass
        return

    def __process_czero(self, x_n=300):
        """
        update process
        """
        try:
            # wait until get camera image
            while MPVariable.lane_m_leasts_abc_lpf_c.value == 0.0:
                time.sleep(1)

            # fig setting
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(6,3), dpi=80)

            X = []
            czeroY = X*0
            c_zero, = plt.plot(X, czeroY, color='Cyan', linewidth=2, label='c_zero')

            ax = fig.add_subplot(111)
            ax.yaxis.tick_right()

            plt.title("c_zero")
            fig.canvas.set_window_title('c_zero')
            plt.xlabel("frame[-]")
            plt.ylabel("c_zero[m]")
            plt.legend(loc='upper left')
            plt.xlim(0, x_n)
            plt.ylim(-1.5, 1.5)

            while self.__is_running.value:
                if len(X) < x_n:
                    X = np.append(X, MPVariable.camera_frame_counter.value)
                    czeroY = np.append(czeroY, MPVariable.lane_m_leasts_abc_lpf_c.value)
                else:
                    # set new data
                    X[0] = MPVariable.camera_frame_counter.value
                    czeroY[0] = MPVariable.lane_m_leasts_abc_lpf_c.value
                    # shift
                    X = np.roll(X, -1)
                    czeroY = np.roll(czeroY, -1)

                # update axis
                c_zero.set_data(X, czeroY)
                plt.xlim(max(X)-x_n, max(X))

                # draw
                plt.pause(.1)

        except KeyboardInterrupt:
            pass
        finally:
            self.__is_running.value = False
            pass
        return


if __name__ == '__main__':
    ymlfile = open('../pilot_config.yml')
    cfg = yaml.load(ymlfile)
    ymlfile.close()

    mpchart = MPChart(cfg)

    try:
        while True:
            MPVariable.camera_frame_counter.value += 1
            MPVariable.can_tx_target_angle.value = np.random.randn()
            MPVariable.can_rx_actual_angle.value = np.random.randn()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass

    mpchart.end()
