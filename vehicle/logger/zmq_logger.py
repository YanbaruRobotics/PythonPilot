#!/usr/bin/python3
# -*- coding: utf-8 -*-

import csv
import cv2
import multiprocessing
import numpy as np
import os
import struct
import sys
import threading
import time
import yaml
import zmq


class ZMQLogger():
    def __init__(self, cfg):
        # Decide output path based on datetime
        start_datetime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        output_dir = cfg['log_output_dir'] + '{:s}/'.format(str(start_datetime))
        os.makedirs(output_dir)

        # Initialize workers
        t_sub_devicerx = threading.Thread(target=self.__worker_sub_csv, \
                                          args=(cfg['zmq_localhost'], \
                                                cfg['zmq_port_pubsub_devicerx'], \
                                                cfg['zmq_topic_devicerx'], \
                                                cfg['zmq_type_list_devicerx'], \
                                                cfg['csv_header_list_devicerx'], \
                                                output_dir))
        t_sub_devicetx = threading.Thread(target=self.__worker_sub_csv, \
                                          args=(cfg['zmq_localhost'], \
                                                cfg['zmq_port_pubsub_devicetx'], \
                                                cfg['zmq_topic_devicetx'], \
                                                cfg['zmq_type_list_devicetx'], \
                                                cfg['csv_header_list_devicetx'], \
                                                output_dir))
        # Start workers
        t_sub_devicerx.setDaemon(True)
        t_sub_devicetx.setDaemon(True)
        t_sub_devicerx.start()
        t_sub_devicetx.start()

        # Initialize process
        self.__m_sub_camera = multiprocessing.Process(target=self.__process_sub_mp4, \
                                                      args=(cfg['zmq_localhost'], \
                                                            cfg['zmq_port_pubsub_camera'], \
                                                            cfg['zmq_topic_camera'], \
                                                            cfg['camera_width'], \
                                                            cfg['camera_height'], \
                                                            cfg['camera_fps'], \
                                                            output_dir))
        # Start process
        self.__m_sub_camera.start()
        return

    def end(self):
        """
        End process
        """
        self.__m_sub_camera.join()
        print('Finish ZMQLogger')
        return

    def __worker_sub_csv(self, address, port, topic, type_list, header_list, output_dir):
        """
        Update process
        """
        try:
            # setting zmq
            context = zmq.Context()
            socket_sub = context.socket(zmq.SUB)
            socket_sub.connect("tcp://{}:{}".format(address, port))
            socket_sub.setsockopt_string(zmq.SUBSCRIBE, topic)

            # setting csv file
            output_path = output_dir + '{}.csv'.format(topic)
            f = open(output_path, 'w')
            writer = csv.writer(f, lineterminator='\n')

            # Prepare store data
            data = np.zeros(len(type_list))

            # Write csv header
            header_list.append('zmq_diff_time_sec')
            writer.writerow(header_list) 

            # Start logging
            while True:
                # receive zmq
                c = socket_sub.recv_multipart()

                # store data except topic
                for i in range(len(c)-1):
                    data[i] = struct.unpack(type_list[i], c[i+1])[0]

                # calc delay
                now_time = time.time()
                diff_time = now_time - data[0] / 1000000.0
                data_with_diff = np.insert(data, len(type_list), diff_time)

                # write to csv
                writer.writerow(data_with_diff)

                # debug
                # print(now_time, topic, 'diff', diff_time)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            f.close()
            socket_sub.close()
            context.destroy()
        return

    def __process_sub_mp4(self, address, port, topic, width, height, fps, output_dir):
        """
        Update process
        """
        try:
            # setting zmq
            context = zmq.Context()
            socket_sub = context.socket(zmq.SUB)
            socket_sub.connect("tcp://{}:{}".format(address, port))
            socket_sub.setsockopt_string(zmq.SUBSCRIBE, topic)

            # setting mp4 file
            output_path = output_dir + '{}.mp4'.format(topic)
            fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                sys.exit()

            # Start logging
            while True:
                # receive zmq
                c, frame_time_byte, frame_cnt_byte, mat_type_byte, img_byte = socket_sub.recv_multipart()

                # Convert byte to integer
                frame_time = struct.unpack('q', frame_time_byte)[0]
                frame_cnt = struct.unpack('q', frame_cnt_byte)[0]
                mat_type = struct.unpack('q', mat_type_byte)[0]

                # calc delay
                now_time = time.time()
                diff_time = now_time - frame_time / 1000000.0

                # calc img
                img_np = np.frombuffer(img_byte, dtype=np.uint8).reshape((height,width,3));

                # save img
                out.write(img_np)

                # debug
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
            cv2.destroyAllWindows()
            out.release()
            socket_sub.close()
            context.destroy()
        return


if __name__ == '__main__':
    ymlfile = open('../config.yml')
    cfg = yaml.load(ymlfile)
    cfg['log_output_dir'] = '../' + cfg['log_output_dir']
    ymlfile.close()

    logger = ZMQLogger(cfg)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    logger.end()
