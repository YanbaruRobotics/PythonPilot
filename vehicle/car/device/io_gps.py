#!/usr/bin/python3
# -*- coding: utf-8 -*-

import ctypes
import multiprocessing
import numpy as np
import time
import yaml
import zmq


class MPGPSRx():
    def __init__(self, cfg):
        self.__is_use_gps_port = cfg['use_gps']

        # Data for store
        self.latitude = multiprocessing.Value(ctypes.c_float,cfg['latitude_init'])
        self.longitude = multiprocessing.Value(ctypes.c_float,cfg['longitude_init'])

        # Initialize process
        self.__m = multiprocessing.Process(target=self.__process_gps, \
                                           args=(cfg['gps_interval'],))
        # Start process
        self.__m.start()
        return

    def end(self):
        """
        End process
        """
        self.__m.join()
        print('Finish MPGPS')
        return

    def __process_gps(self, interval):
        """
        Connect to GPS port
        """
        if self.__is_use_gps_port:
            pass

        """
        Update GPS data
        """
        try:
            last_get_time = time.time()
            while True:
                now_time = time.time()
                if (now_time - last_get_time) >= interval:
                    # update value
                    if self.__is_use_gps_port:
                        self.latitude.value, self.longitude.value = 123.0, 456.0
                    # store time
                    last_get_time = now_time
                time.sleep(interval / 10.0)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            # Close GPS port
            if self.__is_use_gps_port:
                pass
            pass
        return


if __name__ == '__main__':
    ymlfile = open('../../vehicle_config.yml')
    cfg = yaml.load(ymlfile)
    ymlfile.close()

    gpsrx = MPGPSRx(cfg)

    try:
        while True:
            print(time.time(), \
                  'GPSRx', \
                  gpsrx.latitude.value, \
                  gpsrx.longitude.value)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    gpsrx.end()
