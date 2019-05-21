#!/usr/bin/python3
# -*- coding: utf-8 -*-

import multiprocessing
import numpy as np
import time

from common.pilot_gloval_variable import MPVariable
from planning import pure_pursuit


class MPPlanning():
    def __init__(self, cfg):
        self.__m = multiprocessing.Process(target=self.__process, \
                                           args=(cfg['planning_interval'],))
        self.__m.start()
        return

    def end(self):
        self.__m.join()
        print('Finish MPPerception')
        return

    def __process(self, interval):
        """
        update planning
        """
        try:
            previos_work_time = time.time()
            while True:
                now_time = time.time()
                if (now_time - previos_work_time) >= interval:
                    # calc target angle
                    target_z, \
                    target_angle = pure_pursuit.pure_pursuit(MPVariable.lane_m_leasts_abc_lpf_a.value, \
                                                             MPVariable.lane_m_leasts_abc_lpf_b.value, \
                                                             MPVariable.lane_m_leasts_abc_lpf_c.value, \
                                                             MPVariable.obd_vehicle_speed_kmph.value)

                    # tx update
                    MPVariable.pp_target_z.value = target_z
                    MPVariable.can_tx_target_angle.value = target_angle
                    MPVariable.can_tx_servo_on_flag.value = MPVariable.lane_m_leasts_status.value
                    MPVariable.can_tx_counter_camera_unit.value = MPVariable.can_rx_counter_servo_unit.value

                    previos_work_time = now_time
        except KeyboardInterrupt:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            pass
        return
