#!/usr/bin/python3
# -*- coding: utf-8 -*-

import queue as Queue
import tensorflow as tf
import threading
import time


class SessionWorker():
    def __init__(self, tag, graph, config, sess=None):
        self.lock = threading.Lock()
        self.sess_queue = Queue.Queue(maxsize=1)
        self.result_queue = Queue.Queue(maxsize=1)
        self.tag = tag

        if sess is not None:
            self.sess = sess
        else:
            graph.as_default()
            self.sess = tf.Session(config=config)

        t = threading.Thread(target=self.execution, args=())
        t.setDaemon(True)
        t.start()
        return

    def execution(self):
        self.is_thread_running = True
        try:
            while self.is_thread_running:
                while not self.sess_queue.empty():
                    q = self.sess_queue.get(block=False)
                    opts = q["opts"]
                    feeds= q["feeds"]
                    extras= q["extras"]
                    if extras is not None:
                        """ add in time """
                        extras.update({self.tag+"_in_time":time.time()})
                    if feeds is None:
                        results = self.sess.run(opts)
                    else:
                        results = self.sess.run(opts, feed_dict=feeds)
                    if extras is not None:
                        """ add out time """
                        extras.update({self.tag+"_out_time":time.time()})
                    self.result_queue.put({"results":results, "extras":extras})
                    self.sess_queue.task_done()
                else:
                    time.sleep(0.001)
            self.sess.close()

        except:
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
            self.sess.close()
        return

    def is_sess_empty(self):
        if self.sess_queue.empty():
            return True
        else:
            return False

    def put_sess_queue(self, opts, feeds=None, extras={}):
        self.sess_queue.put({"opts":opts, "feeds":feeds, "extras":extras})
        return

    def is_result_empty(self):
        if self.result_queue.empty():
            return True
        else:
            return False

    def get_result_queue(self):
        result = None
        if not self.result_queue.empty():
            result = self.result_queue.get(block=False)
            self.result_queue.task_done()
        return result

    def stop(self):
        self.is_thread_running=False
        with self.lock:
            while not self.sess_queue.empty():
                q = self.sess_queue.get(block=False)
                self.sess_queue.task_done()
        return
