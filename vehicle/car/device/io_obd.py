#!/usr/bin/python3
# -*- coding: utf-8 -*-

import ctypes
import multiprocessing
import numpy as np
import obd
import time
import yaml
import zmq


class MPOBDRx():
    def __init__(self, cfg):
        self.__is_use_obd_port = cfg['use_obd']

        # Data for store
        self.temp = multiprocessing.Value(ctypes.c_int,0)
        self.speed = multiprocessing.Value(ctypes.c_int,0)
        self.rpm = multiprocessing.Value(ctypes.c_float,0.0)

        # Initialize process
        self.__m = multiprocessing.Process(target=self.__process_obd, \
                                           args=(cfg['obd_protocol'], \
                                                 cfg['obd_interval']))
        # Start process
        self.__m.start()
        return

    def end(self):
        """
        End process
        """
        self.__m.join()
        print('Finish MPOBD')
        return

    def __process_obd(self, protocol, interval):
        """
        Connect to OBDII adapter
        """
        while self.__is_use_obd_port:
            try:
                ports = obd.scan_serial()
                connection = obd.OBD(ports[0], protocol=protocol)
                print('OBD connection status:', connection.status())
                break
            except Exception as e:
                time.sleep(10)

        """
        Update process
        """
        try:
            last_get_time = time.time()
            while True:
                now_time = time.time()
                if (now_time - last_get_time) >= interval:
                    if self.__is_use_obd_port:
                        """
                        Get OBD data
                        """
                        try:
                            self.temp.value = int(connection.query(obd.commands.COOLANT_TEMP).value.magnitude)
                            self.speed.value = int(connection.query(obd.commands.SPEED).value.magnitude)
                            self.rpm.value = float(connection.query(obd.commands.RPM).value.magnitude)
                        except Exception as e:
                            pass
                    # store time
                    last_get_time = now_time
                time.sleep(interval / 10.0)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            if self.__is_use_obd_port:
                connection.close()
        return


if __name__ == '__main__':
    ymlfile = open('../../vehicle_config.yml')
    cfg = yaml.load(ymlfile)
    ymlfile.close()

    obdrx = MPOBDRx(cfg)

    try:
        while True:
            print(time.time(), \
                  'OBDRx', \
                  obdrx.temp.value, \
                  obdrx.speed.value, \
                  obdrx.rpm.value)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    obdrx.end()
