#!/usr/bin/python3
# -*- coding: utf-8 -*-

import ctypes
import cv2
import multiprocessing
import numpy as np
import os
import queue as Queue
import struct
import sys
import threading
import time
import yaml
import zmq

sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))

from common.pilot_gloval_variable import MPVariable
from common.camera_img_pipe import start_sender


class SubDeviceCamera():
    def __init__(self, cfg):
        # Initialize workers
        t_sub_devicerx = threading.Thread(target=self.__worker_sub_device, \
                                          args=(cfg['zmq_localhost'], \
                                                cfg['zmq_port_pubsub_devicerx'], \
                                                cfg['zmq_topic_devicerx'], \
                                                cfg['zmq_type_list_devicerx']))
        # Start workers
        t_sub_devicerx.setDaemon(True)
        t_sub_devicerx.start()

        # Initialize process
        self.__m_sub_camera = multiprocessing.Process(target=self.__process_sub_camera, \
                                                      args=(cfg['zmq_localhost'], \
                                                            cfg['zmq_port_pubsub_camera'], \
                                                            cfg['zmq_topic_camera'], \
                                                            cfg['camera_width'], \
                                                            cfg['camera_height']))
        # Start process
        self.__m_sub_camera.start()
        return

    def end(self):
        """
        End process
        """
        # Close process
        self.__m_sub_camera.join()
        print('Finish SubDeviceCamera')
        return

    def __worker_sub_device(self, address, port, topic, type_list):
        """
        Update process
        """
        try:
            # setting zmq
            context = zmq.Context()
            socket_sub = context.socket(zmq.SUB)
            socket_sub.connect("tcp://{}:{}".format(address, port))
            socket_sub.setsockopt_string(zmq.SUBSCRIBE, topic)

            # Prepare store data
            data = [0] * len(type_list)

            # Start logging
            while True:
                # receive zmq
                c = socket_sub.recv_multipart()

                # convert data
                for i in range(len(c)-1):
                    data[i] = struct.unpack(type_list[i], c[i+1])[0]
                    # print(i, type_list[i], data[i])

                # msg head
                published_time = data[0]
                camera_frame_counter = data[1]

                # msg can rx
                MPVariable.can_rx_counter_servo_unit.value = data[2]
                MPVariable.can_rx_time_us_diff.value = data[3]
                MPVariable.can_rx_button_y.value = data[4]
                MPVariable.can_rx_button_g.value = data[5]
                MPVariable.can_rx_button_r.value = data[6]
                MPVariable.can_rx_actual_angle.value = data[7]
                MPVariable.can_error_count_rx.value = data[8]
                MPVariable.can_error_count_tx.value = data[9]

                # msg obd rx
                MPVariable.obd_temp_degrees_celsius.value = data[10]
                MPVariable.obd_vehicle_speed_kmph.value = data[11]
                MPVariable.obd_engine_speed_rpm.value = data[12]

                # msg gps rx
                MPVariable.gps_latitude.value = data[13]
                MPVariable.gps_longitude.value = data[14]

                # calc delay
                now_time = time.time()
                diff_time = now_time - published_time / 1000000.0

                # debug
                # print(now_time, topic, 'diff', diff_time)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            socket_sub.close()
            context.destroy()
        return

    def __process_sub_camera(self, address, port, topic, width, height):
        """
        Update process
        """
        try:
            # setting zmq
            context = zmq.Context()
            socket_sub = context.socket(zmq.SUB)
            socket_sub.connect("tcp://{}:{}".format(address, port))
            socket_sub.setsockopt_string(zmq.SUBSCRIBE, topic)

            # setting img pipe
            q_out = Queue.Queue()
            start_sender(MPVariable.zmq_out_con, q_out)

            # Start logging
            while True:
                # receive zmq
                c, frame_time_byte, frame_cnt_byte, mat_type_byte, img_byte = socket_sub.recv_multipart()

                # Convert byte to integer
                frame_time = struct.unpack('q', frame_time_byte)[0]
                frame_cnt = struct.unpack('q', frame_cnt_byte)[0]
                mat_type = struct.unpack('q', mat_type_byte)[0]

                # Store data to gloval
                MPVariable.camera_frame_counter.value = frame_cnt

                # calc delay
                now_time = time.time()
                diff_time = now_time - frame_time / 1000000.0

                # calc img
                img_np = np.frombuffer(img_byte, dtype=np.uint8).reshape((height,width,3))

                q_out.put({'image': img_np})

                # # debug
                # print(now_time, topic, 'diff', diff_time)
                # # show img
                # cv2.imshow('img', img_np)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

        except KeyboardInterrupt:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            q_out.put(None)
            cv2.destroyAllWindows()
            socket_sub.close()
            context.destroy()
        return


if __name__ == '__main__':
    ymlfile = open('../../vehicle/vehicle_config.yml')
    cfg = yaml.load(ymlfile)
    ymlfile.close()

    receiver = SubDeviceCamera(cfg)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    receiver.end()
