#!/usr/bin/python3
# -*- coding: utf-8 -*-

import can
import cantools
import ctypes
import multiprocessing
import time
import yaml


class MPCANRx():
    def __init__(self, cfg):
        self.__is_use_can_port = cfg['use_can']

        # Data for store
        self.rx_counter_servo_unit = multiprocessing.Value(ctypes.c_int,0)
        self.rx_time_us_diff = multiprocessing.Value(ctypes.c_int,0)
        self.rx_button_y = multiprocessing.Value(ctypes.c_bool,False)
        self.rx_button_g = multiprocessing.Value(ctypes.c_bool,False)
        self.rx_button_r = multiprocessing.Value(ctypes.c_bool,False)
        self.rx_actual_angle = multiprocessing.Value(ctypes.c_float,0.0)
        self.can_error_count_rx = multiprocessing.Value(ctypes.c_int,0)

        # Initialize process
        self.__m = multiprocessing.Process(target=self.__process_can, \
                                           args=(cfg['can_name'], \
                                                 cfg['can_bustype'], \
                                                 cfg['can_bitrate'], \
                                                 cfg['can_dbc_path'], \
                                                 cfg['can_rx_interval']))
        # Start process
        self.__m.start()

        return

    def end(self):
        """
        End process
        """
        self.__m.join()
        print('Finish CANRx')
        return

    def __process_can(self, canname, bustype, bitrate, can_dbc_path, interval):
        """
        Connect to CAN port
        """
        while self.__is_use_can_port:
            try:
                can_bus = can.interface.Bus(canname, bustype=bustype, bitrate=bitrate)
                print('Connected CANRx')
                break
            except Exception as e:
                time.sleep(10)

        """
        Load CAN database
        """
        can_db = cantools.database.load_file(can_dbc_path)
        message_servounit_to_camera = can_db.get_message_by_name('ServoUnitToCamera')

        """
        Update CAN Rx data
        """
        try:
            while True:
                if self.__is_use_can_port:
                    # Receive CAN data
                    try:
                        message_rx = can_bus.recv(timeout=0.100)
                    except Exception as e:
                        print(e)
                        self.can_error_count_rx.value += 1
                    # Decode CAN data
                    if message_rx is not None:
                        if message_rx.arbitration_id == message_servounit_to_camera.frame_id:
                            data_rx = can_db.decode_message(message_rx.arbitration_id, message_rx.data)
                            if data_rx:
                                self.rx_counter_servo_unit.value = data_rx['counter_servo_unit']
                                self.rx_time_us_diff.value = data_rx['time_us_diff']
                                self.rx_button_y.value = data_rx['button_y']
                                self.rx_button_g.value = data_rx['button_g']
                                self.rx_button_r.value = data_rx['button_r']
                                self.rx_actual_angle.value = data_rx['actual_angle']
                else:
                    time.sleep(interval)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            if self.__is_use_can_port:
                can_bus.shutdown()
        return


class MPCANTx():
    def __init__(self, cfg):
        self.__is_use_can_port = cfg['use_can']

        # Data for store
        self.tx_counter_camera_unit = multiprocessing.Value(ctypes.c_int,0)
        self.tx_servo_on_flag = multiprocessing.Value(ctypes.c_bool,False)
        self.tx_target_angle = multiprocessing.Value(ctypes.c_float,0.0)
        self.can_error_count_tx = multiprocessing.Value(ctypes.c_int,0)

        # Initialize process
        self.__m = multiprocessing.Process(target=self.__update, \
                                           args=(cfg['can_name'], \
                                                 cfg['can_bustype'], \
                                                 cfg['can_bitrate'], \
                                                 cfg['can_dbc_path'], \
                                                 cfg['can_tx_interval']))
        # Start process
        self.__m.start()

        return

    def end(self):
        """
        End process
        """
        self.__m.join()
        print('Finish CANTx')
        return

    def __update(self, canname, bustype, bitrate, can_dbc_path, interval):
        """
        Connect to CAN port
        """
        while self.__is_use_can_port:
            try:
                can_bus = can.interface.Bus(canname, bustype=bustype, bitrate=bitrate)
                print('Connected CANTx')
                break
            except Exception as e:
                time.sleep(10)

        """
        Load CAN database
        """
        can_db = cantools.database.load_file(can_dbc_path)
        message_camera_to_servounit = can_db.get_message_by_name('CameraToServoUnit')

        """
        Update CAN Tx data
        """
        try:
            last_send_time = time.time()
            # succsess_cnt = 0
            while True:
                now_time = time.time()
                if self.__is_use_can_port and (now_time - last_send_time) >= interval:
                    # Encode CAN data
                    can_data_tx = message_camera_to_servounit.encode({'counter_camera_unit': self.tx_counter_camera_unit.value, \
                                                                      'servo_on_flag': self.tx_servo_on_flag.value, \
                                                                      'target_angle': self.tx_target_angle.value})
                    message_tx = can.Message(arbitration_id=message_camera_to_servounit.frame_id, data=can_data_tx)
                    # Send encoded CAN data
                    try:
                        can_bus.send(message_tx, timeout=0.010)
                        # succsess_cnt += 1
                        # print(succsess_cnt)
                        last_send_time = time.time()
                        time.sleep(interval/10)
                    except Exception as e:
                        print('CANTx', e)
                        # ToDo: Fix can tx error
                        tx_error_time = time.time()
                        can_bus.shutdown()
                        can_bus = can.interface.Bus(canname, bustype=bustype, bitrate=bitrate)
                        print('CANTx Restart', time.time() - tx_error_time)
                        self.can_error_count_tx.value += 1
                else:
                    time.sleep(interval)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            if self.__is_use_can_port:
                can_bus.shutdown()
            pass
        return


if __name__ == '__main__':
    ymlfile = open('../../vehicle_config.yml')
    cfg = yaml.load(ymlfile)
    cfg['can_dbc_path'] = '../../' + cfg['can_dbc_path']
    ymlfile.close()

    canrx = MPCANRx(cfg)
    cantx = MPCANTx(cfg)

    try:
        while True:
            print(time.time(), \
                  'CANRx', \
                  canrx.rx_counter_servo_unit.value, \
                  canrx.rx_time_us_diff.value, \
                  canrx.rx_button_y.value, \
                  canrx.rx_button_g.value, \
                  canrx.rx_button_r.value, \
                  canrx.rx_actual_angle.value, \
                  canrx.can_error_count_rx.value)

            cantx.tx_target_angle.value += 0.01
            print(time.time(), \
                  'CANTx', \
                  cantx.tx_counter_camera_unit.value, \
                  cantx.tx_servo_on_flag.value, \
                  cantx.tx_target_angle.value, \
                  cantx.can_error_count_tx.value)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    canrx.end()
    cantx.end()
