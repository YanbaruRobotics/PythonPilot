#!/usr/bin/python3
# -*- coding: utf-8 -*-

import ctypes
import multiprocessing


class MPIOVariable():
    """
    SHARED VARIABLES IN MULTIPROSESSING
    """
    camera_frame_counter = multiprocessing.Value(ctypes.c_longlong,0)
