#!/usr/bin/python3
# -*- coding: utf-8 -*-

import threading
import time


def start_receiver(in_con, q, mp_drop_frames):
    """
    START THREAD
    """
    t = threading.Thread(target=receive, args=(in_con, q, mp_drop_frames,))
    t.setDaemon(True)
    t.start()
    return t

def receive(in_con, q, mp_drop_frames):
    """
    READ CONNECTION TO QUEUE
    """
    while True:
        data = in_con.recv()
        if data is None:
            q.put(data)
            break
        if q.empty():
            q.put(data)
        else:
            mp_drop_frames.value += 1
    in_con.close()
    return

def start_sender(out_con, q):
    """
    START THREAD
    """
    t = threading.Thread(target=send, args=(out_con, q, ))
    t.setDaemon(True)
    t.start()
    return t

def send(out_con, q):
    """
    READ QUEUE TO CONNECTION
    """
    while True:
        data = q.get(block=True)
        q.task_done()
        out_con.send(data)
        if data is None:
            break
    out_con.close()
    return
