#!/usr/bin/python3
# -*- coding: utf-8 -*-

import ctypes
import multiprocessing


class MPVariable():
    """
    For ZMQ subscriber
    """
    camera_frame_counter = multiprocessing.Value(ctypes.c_longlong,0)

    """ CAN """
    can_rx_counter_servo_unit = multiprocessing.Value(ctypes.c_int,0)
    can_rx_time_us_diff = multiprocessing.Value(ctypes.c_int,0)
    can_rx_button_y = multiprocessing.Value(ctypes.c_bool,False)
    can_rx_button_g = multiprocessing.Value(ctypes.c_bool,False)
    can_rx_button_r = multiprocessing.Value(ctypes.c_bool,False)
    can_rx_actual_angle = multiprocessing.Value(ctypes.c_float,0.0)

    can_error_count_rx = multiprocessing.Value(ctypes.c_int,0)
    can_error_count_tx = multiprocessing.Value(ctypes.c_int,0)

    """ OBD """
    obd_temp_degrees_celsius = multiprocessing.Value(ctypes.c_int,0)
    obd_vehicle_speed_kmph = multiprocessing.Value(ctypes.c_int,0)
    obd_engine_speed_rpm = multiprocessing.Value(ctypes.c_float,0.0)

    """ GPS """
    gps_latitude = multiprocessing.Value(ctypes.c_float,0.0)
    gps_longitude = multiprocessing.Value(ctypes.c_float,0.0)


    """
    For ZMQ publisher
    """
    can_tx_counter_camera_unit = multiprocessing.Value(ctypes.c_int,0)
    can_tx_servo_on_flag = multiprocessing.Value(ctypes.c_bool,False)
    can_tx_target_angle = multiprocessing.Value(ctypes.c_float,0.0)


    """
    For Perception
    """
    """ lane """
    lane_m_leasts_status = multiprocessing.Value(ctypes.c_bool,False)
    lane_m_leasts_abc_lpf_a = multiprocessing.Value(ctypes.c_float,0.0)
    lane_m_leasts_abc_lpf_b = multiprocessing.Value(ctypes.c_float,0.0)
    lane_m_leasts_abc_lpf_c = multiprocessing.Value(ctypes.c_float,0.0)
    lane_m_width_lpf = multiprocessing.Value(ctypes.c_float,0.0)

    lane_l_leasts_status = multiprocessing.Value(ctypes.c_bool,False)
    lane_l_leasts_abc_lpf_a = multiprocessing.Value(ctypes.c_float,0.0)
    lane_l_leasts_abc_lpf_b = multiprocessing.Value(ctypes.c_float,0.0)
    lane_l_leasts_abc_lpf_c = multiprocessing.Value(ctypes.c_float,0.0)
    lane_l_leasts_z_max = multiprocessing.Value(ctypes.c_float,0.0)

    lane_r_leasts_status = multiprocessing.Value(ctypes.c_bool,False)
    lane_r_leasts_abc_lpf_a = multiprocessing.Value(ctypes.c_float,0.0)
    lane_r_leasts_abc_lpf_b = multiprocessing.Value(ctypes.c_float,0.0)
    lane_r_leasts_abc_lpf_c = multiprocessing.Value(ctypes.c_float,0.0)
    lane_r_leasts_z_max = multiprocessing.Value(ctypes.c_float,0.0)


    """
    For Planning
    """
    pp_target_z = multiprocessing.Value(ctypes.c_float,0.0)


    """
    MULTI-PROCESSING PIPE
    """
    det_in_con, zmq_out_con = multiprocessing.Pipe(duplex=False)
    vis_in_con, det_out_con = multiprocessing.Pipe(duplex=False)
    bv_in_con, det_out_bv_con = multiprocessing.Pipe(duplex=False)

    det_drop_frames = multiprocessing.Value(ctypes.c_int,0)
    vis_drop_frames = multiprocessing.Value(ctypes.c_int,0)
    bv_drop_frames = multiprocessing.Value(ctypes.c_int,0)


    """
    VISUALIZATION
    """
    perception_object_fps = multiprocessing.Value(ctypes.c_int,0)
    perception_lane_fps = multiprocessing.Value(ctypes.c_int,0)


    """
    CLOSE MULTI-PROCESSING PIPE
    """
    def __del__(self):
        zmq_out_con.close()
        det_in_con.close()
        det_out_con.close()
        vis_in_con.close()
        det_out_bv_con.close()
        bv_in_con.close()
