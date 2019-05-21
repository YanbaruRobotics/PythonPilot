#!/usr/bin/python3
# -*- coding: utf-8 -*-

import csv
import ctypes
import cv2
import multiprocessing
import numpy as np
import threading
import time
import yaml
import zmq


class ZMQUnlogger():
    def __init__(self, cfg):
        # Bridge camera_frame_counter from mp4 process to csv worker
        self.__camera_frame_counter = multiprocessing.Value(ctypes.c_int,0)
        self.__is_using_csv = cfg['use_csv']

        # Initialize workerss
        if self.__is_using_csv:
            t_pub_devicerx = threading.Thread(target=self.__worker_pub_csv, \
                                              args=(cfg['zmq_localhost'], \
                                                    cfg['zmq_port_pubsub_devicerx'], \
                                                    cfg['zmq_topic_devicerx'], \
                                                    cfg['zmq_type_list_devicerx'], \
                                                    cfg['camera_fps'], \
                                                    cfg['unlogger_shift_sec'], \
                                                    cfg['unlogger_path']))
            # Start workers
            t_pub_devicerx.setDaemon(True)
            t_pub_devicerx.start()

        # Initialize process
        self.__m_pub_camera = multiprocessing.Process(target=self.__process_pub_mp4, \
                                                      args=(cfg['zmq_localhost'], \
                                                            cfg['zmq_port_pubsub_camera'], \
                                                            cfg['zmq_topic_camera'], \
                                                            cfg['camera_fps'], \
                                                            cfg['unlogger_fps'], \
                                                            cfg['unlogger_shift_sec'], \
                                                            cfg['unlogger_path']))
        # Start process
        self.__m_pub_camera.start()
        return

    def end(self):
        """
        End process
        """
        self.__m_pub_camera.join()
        print('Finish ZMQUnlogger')
        return

    def __worker_pub_csv(self, address, port, topic, type_list, camera_fps, shift_sec, input_dir):
        """
        Update process
        """
        try:
            # Initialize zmq socket
            context = zmq.Context()
            socket = context.socket(zmq.PUB)
            socket.bind("tcp://{}:{}".format(address, port))

            # Setting path
            input_path = input_dir + '{}.csv'.format(topic)

            # Open csv file
            f = open(input_path, 'r')
            reader = csv.reader(f)
            header = next(reader)

            # Data for store
            frame_csv = 0

            while True:
                # Sync camera frame
                if frame_csv >= self.__camera_frame_counter.value:
                    if frame_csv >= int(shift_sec*camera_fps):
                        time.sleep(0.001)
                    continue

                # Read next csv row
                try:
                    data_list_s = next(reader)
                except Exception as e:
                    break

                # pack
                data_list_np = []
                for i in range(len(type_list)):
                    value_f = float(data_list_s[i])
                    if type_list[i] == 'q':
                        data_list_np.append(np.array([int(value_f)]))
                    elif type_list[i] == '?':
                        data_list_np.append(np.array([bool(value_f)]))
                    else:
                        data_list_np.append(np.array([value_f]))

                # Update frame_csv
                frame_csv = int(data_list_np[1])

                # send msg
                if frame_csv >= int(shift_sec*camera_fps):
                    msg_topic = [topic.encode('utf-8')]
                    msg_data = data_list_np[:len(data_list_np)-1]   # except diff time
                    msg = msg_topic + msg_data
                    socket.send_multipart(msg)

                # Debug
                # print(topic, frame_csv, msg)
                # print('\n')

        except KeyboardInterrupt:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            f.close()
        return


    def __process_pub_mp4(self, address, port, topic, camera_fps, unlogger_fps, shift_sec, input_dir):
        """
        Update process
        """
        try:
            # Initialize zmq socket
            context = zmq.Context()
            socket = context.socket(zmq.PUB)
            socket.bind("tcp://{}:{}".format(address, port))

            # Setting video path
            input_path = input_dir + '{}.mp4'.format(topic)

            # Initialize cv2 capture
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise IOError(("Couldn't open video file."))

            # Skip frame
            cap.set(1, shift_sec*camera_fps)
            self.__camera_frame_counter.value += int(shift_sec*camera_fps)

            # Setting fps limit
            interval = 1.0 / unlogger_fps
            previos_work_time = time.time()

            # main loop
            while True:
                now_time = time.time()
                if now_time >= previos_work_time + interval:
                    # update frame
                    ret, img = cap.read()
                    if not ret:
                        break

                    # store msg
                    ndim = img.ndim
                    msg = [topic.encode('utf-8'), 
                           np.array([int(time.time() * 1000 * 1000)]), \
                           np.array([self.__camera_frame_counter.value]), \
                           np.array([ndim]), \
                           img.data]

                    # send msg
                    socket.send_multipart(msg)

                    # count up
                    self.__camera_frame_counter.value += 1

                    # for unit test debug
                    # show img
                    # cv2.imshow('img', img)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break

                    # store time
                    previos_work_time = now_time
                time.sleep(interval/10)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()
            cap.release()
            socket.close()
            context.destroy()
        return


if __name__ == '__main__':
    ymlfile = open('../vehicle_config.yml')
    cfg = yaml.load(ymlfile)
    cfg['unlogger_path'] = '../' + cfg['unlogger_path']
    ymlfile.close()

    unlogger = ZMQUnlogger(cfg)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    unlogger.end()
