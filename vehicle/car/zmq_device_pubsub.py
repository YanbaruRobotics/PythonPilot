#!/usr/bin/python3
# -*- coding: utf-8 -*-

import ctypes
import multiprocessing
import numpy as np
import os
import struct
import sys
import time
import yaml
import zmq

sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))

from car.car_gloval_variable import MPIOVariable
from car.device.io_can import MPCANRx, MPCANTx
from car.device.io_obd import MPOBDRx
from car.device.io_gps import MPGPSRx


class PubSubDevice():
    def __init__(self, cfg):
        # Bridge Tx data from subscribe process to publish process
        self.__can_error_count_tx = multiprocessing.Value(ctypes.c_int,0)

        # Initialize process
        self.__m_pub = multiprocessing.Process(target=self.__process_pub, \
                                               args=(cfg,
                                                     cfg['zmq_localhost'], \
                                                     cfg['zmq_port_pubsub_devicerx'], \
                                                     cfg['zmq_topic_devicerx'], \
                                                     cfg['zmq_interval_devicerx']))
        self.__m_sub = multiprocessing.Process(target=self.__process_sub, \
                                               args=(cfg,
                                                     cfg['zmq_localhost'], \
                                                     cfg['zmq_port_pubsub_devicetx'], \
                                                     cfg['zmq_topic_devicetx'], \
                                                     cfg['zmq_interval_devicetx']))
        # Start process
        self.__m_pub.start()
        self.__m_sub.start()

        return

    def end(self):
        """
        End process
        """
        self.__m_pub.join()
        self.__m_sub.join()
        print('Finish PubSubData')
        return

    def __process_pub(self, cfg, address, port, topic, interval):
        """
        Update process
        """
        try:
            # Set and start device instance for publishing
            canrx = MPCANRx(cfg)
            obdrx = MPOBDRx(cfg)
            gpsrx = MPGPSRx(cfg)

            # Initialize zmq socket
            context = zmq.Context()
            socket_pub = context.socket(zmq.PUB)
            socket_pub.bind("tcp://{}:{}".format(address, port))

            previos_work_time = time.time()

            # Start publishing
            while True:
                now_time = time.time()
                if (now_time - previos_work_time) >= interval:
                    # prepare msg
                    msg_topic = [topic.encode('utf-8')]

                    msg_head = [np.array([int(now_time * 1000 * 1000)]), \
                                np.array([MPIOVariable.camera_frame_counter.value])]

                    msg_canrx = [np.array([canrx.rx_counter_servo_unit.value]), \
                                 np.array([canrx.rx_time_us_diff.value]), \
                                 np.array([canrx.rx_button_y.value]), \
                                 np.array([canrx.rx_button_g.value]), \
                                 np.array([canrx.rx_button_r.value]), \
                                 np.array([canrx.rx_actual_angle.value]), \
                                 np.array([canrx.can_error_count_rx.value]), \
                                 np.array([self.__can_error_count_tx.value])]

                    msg_obdrx = [np.array([obdrx.temp.value]), \
                                 np.array([obdrx.speed.value]), \
                                 np.array([obdrx.rpm.value])]

                    msg_gpsrx = [np.array([gpsrx.latitude.value]), \
                                 np.array([gpsrx.longitude.value])]

                    msg = msg_topic + msg_head + msg_canrx + msg_obdrx + msg_gpsrx

                    # send msg to zmq
                    socket_pub.send_multipart(msg)

                    # For debug
                    # print(msg)

                    # store time
                    previos_work_time = now_time
                time.sleep(interval / 10.0)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            canrx.end()
            obdrx.end()
            gpsrx.end()
            socket_pub.close()
            context.destroy()
        return

    def __process_sub(self, cfg, address, port, topic, interval):
        """
        Update process
        """
        try:
            # Set and start device instance for publishing
            cantx = MPCANTx(cfg)

            # Initialize zmq socket
            context = zmq.Context()
            socket_sub = context.socket(zmq.SUB)
            socket_sub.connect("tcp://{}:{}".format(address, port))
            socket_sub.setsockopt_string(zmq.SUBSCRIBE, topic)

            # Start subscribing
            while True:
                # Receive zmq socket
                # ToDo: add time out
                c = socket_sub.recv_multipart()
                topic, \
                pub_time_byte, \
                pub_frame_byte, \
                tx_counter_camera_unit_byte, \
                tx_servo_on_flag_byte, \
                tx_target_angle_byte = c

                # Convert byte data
                pub_time = struct.unpack('q', pub_time_byte)[0]
                pub_frame = struct.unpack('q', pub_frame_byte)[0]
                tx_counter_camera_unit = struct.unpack('q', tx_counter_camera_unit_byte)[0]
                tx_servo_on_flag = struct.unpack('?', tx_servo_on_flag_byte)[0]
                tx_target_angle = struct.unpack('d', tx_target_angle_byte)[0]

                # Update CAN Tx data
                cantx.tx_counter_camera_unit.value = tx_counter_camera_unit
                cantx.tx_servo_on_flag.value = tx_servo_on_flag
                cantx.tx_target_angle.value = tx_target_angle

                # Get CAN Tx error
                self.__can_error_count_tx.value = cantx.can_error_count_tx.value

                # For debug
                # print(time.time(), \
                #       'CANTx', \
                #       tx_counter_camera_unit, \
                #       tx_servo_on_flag, \
                #       tx_target_angle)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            cantx.end()
            socket_sub.close()
            context.destroy()
        return


if __name__ == '__main__':
    ymlfile = open('../vehicle_config.yml')
    cfg = yaml.load(ymlfile)
    cfg['can_dbc_path'] = '../' + cfg['can_dbc_path']
    ymlfile.close()

    pubsub_device = PubSubDevice(cfg)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    pubsub_device.end()
