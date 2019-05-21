#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import struct
import sys
import threading
import time
import yaml
import zmq

sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))

from common.pilot_gloval_variable import MPVariable


class PubDevice():
    def __init__(self, cfg):
        # Initialize workers
        t_pub_devicetx = threading.Thread(target=self.__worker_pub_devicetx, \
                                          args=(cfg['zmq_localhost'], \
                                                cfg['zmq_port_pubsub_devicetx'], \
                                                cfg['zmq_topic_devicetx'], \
                                                cfg['zmq_interval_devicetx']))
        # Start workers
        t_pub_devicetx.setDaemon(True)
        t_pub_devicetx.start()

    def end(self):
        return

    def __worker_pub_devicetx(self, address, port, topic, interval):
        """
        Update process
        """
        try:
            # setting zmq
            context = zmq.Context()
            socket_pub = context.socket(zmq.PUB)
            socket_pub.bind("tcp://{}:{}".format(address, port))

            previos_work_time = time.time()

            # Start publish
            while True:
                now_time = time.time()
                if (now_time - previos_work_time) >= interval:
                    msg = [topic.encode('utf-8'), 
                           np.array([int(time.time() * 1000 * 1000)]), \
                           np.array([MPVariable.camera_frame_counter.value]), \
                           np.array([MPVariable.can_tx_counter_camera_unit.value]), \
                           np.array([MPVariable.can_tx_servo_on_flag.value]), \
                           np.array([MPVariable.can_tx_target_angle.value])]
                    socket_pub.send_multipart(msg)

                    # For debug
                    # print(MPVariable.camera_frame_counter.value)

                    # store time
                    previos_work_time = now_time
                time.sleep(interval / 10.0)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            socket_pub.close()
            context.destroy()
        return

if __name__ == '__main__':
    # Load config
    ymlfile = open('../../vehicle/vehicle_config.yml')
    cfg = yaml.load(ymlfile)
    ymlfile.close()

    pub_device = PubDevice(cfg)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    pub_device.end()
