#!/usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
import multiprocessing
import numpy as np
import os
import sys
import time
import yaml
import zmq

sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))

from car.car_gloval_variable import MPIOVariable


class PubCamera():
    def __init__(self, cfg):
        # Initialize process
        self.__m_pub = multiprocessing.Process(target=self.__process_pub, \
                                               args=(cfg['zmq_localhost'], \
                                                     cfg['zmq_port_pubsub_camera'], \
                                                     cfg['zmq_topic_camera'], \
                                                     cfg['camera_input'], \
                                                     cfg['camera_width'], \
                                                     cfg['camera_height'], \
                                                     cfg['camera_fps']))
        # Start process
        self.__m_pub.start()
        return

    def end(self):
        """
        End process
        """
        self.__m_pub.join()
        print('Finish PubCamera')
        return

    def __process_pub(self, address, port, topic, camera_input, width, height, camera_fps):
        """
        Update process
        """
        try:
            # Initialize zmq socket
            context = zmq.Context()
            socket = context.socket(zmq.PUB)
            socket.bind("tcp://{}:{}".format(address, port))

            # Initialize cv2 capture
            cap = cv2.VideoCapture(camera_input)
            if not cap.isOpened():
                raise IOError(("Couldn't open webcam."))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print('PubCamera FPS:', cap.get(cv2.CAP_PROP_FPS))

            # For measuring fps
            fps_check_interval = 10
            last_fps_checked_time = time.time()
            last_fps_checked_frame = 0

            last_send_img_time = time.time()

            while True:
                now_time = time.time()

                # fps limit
                if now_time - last_send_img_time < 1.0 / camera_fps:
                    continue

                # update frame
                ret, img = cap.read()
                if not ret:
                    break

                # store msg
                ndim = img.ndim
                msg = [topic.encode('utf-8'), 
                       np.array([int(time.time() * 1000 * 1000)]), \
                       np.array([MPIOVariable.camera_frame_counter.value]), \
                       np.array([ndim]), \
                       img.data]

                # send msg
                socket.send_multipart(msg)
                last_send_img_time = now_time

                # check fps
                now_time = time.time()
                if now_time - last_fps_checked_time > fps_check_interval:
                    last_fps_checked_time = now_time
                    measured_fps = (MPIOVariable.camera_frame_counter.value - last_fps_checked_frame) / fps_check_interval
                    print('PubCamera measured FPS:', measured_fps)
                    last_fps_checked_frame = MPIOVariable.camera_frame_counter.value

                # count up
                MPIOVariable.camera_frame_counter.value += 1
        except KeyboardInterrupt:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            socket.close()
            context.destroy()
        return


if __name__ == '__main__':
    ymlfile = open('../vehicle_config.yml')
    cfg = yaml.load(ymlfile)
    ymlfile.close()

    pub_camera = PubCamera(cfg)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    pub_camera.end()
