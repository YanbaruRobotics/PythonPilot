#!/usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
import multiprocessing
import numpy as np
import queue as Queue
import tensorflow as tf
import time
import yaml

from common.pilot_gloval_variable import MPVariable
from common.camera_img_pipe import start_receiver, start_sender

from perception.worker_session import SessionWorker

from perception.object_detection.load_graph_nms_v2 import LoadFrozenGraph

from perception.lane_segmentation.dnn_based import LaneNet
from perception.lane_clustering import DbscanBased
from perception.lane_fitting.leasts_based import LeastsBased
from perception.perception_common import handling_image as himg


class MPPerception():
    def __init__(self, cfg):
        self.cfg = cfg
        return

    def end(self):
        print('Finish MPPerception')
        return

    def start(self):
        """ """ """ """ """ """ """ """ """ """ """
        GET CONFIG
        """ """ """ """ """ """ """ """ """ """ """
        FORCE_GPU_COMPATIBLE = self.cfg['force_gpu_compatible']
        LOG_DEVICE           = self.cfg['log_device']
        ALLOW_MEMORY_GROWTH  = self.cfg['allow_memory_growth']
        SPLIT_SHAPE          = self.cfg['split_shape']
        WEIGHTS_PATH         = self.cfg['lane_weights_path']
        object_clip_window_distant = [256, 512]
        """ """

        """ """ """ """ """ """ """ """ """ """ """
        LOAD FROZEN_GRAPH
        """ """ """ """ """ """ """ """ """ """ """
        load_frozen_graph = LoadFrozenGraph(self.cfg)
        graph = load_frozen_graph.load_graph()
        """ """

        """ """ """ """ """ """ """ """ """ """ """
        PREPARE TF CONFIG OPTION
        """ """ """ """ """ """ """ """ """ """ """
        # Session Config: allow seperate GPU/CPU adressing and limit memory allocation
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=LOG_DEVICE)
        config.gpu_options.allow_growth = ALLOW_MEMORY_GROWTH
        config.gpu_options.force_gpu_compatible = FORCE_GPU_COMPATIBLE
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        """ """

        """ """ """ """ """ """ """ """ """ """ """
        LANE DETECTION INSTANCE
        """ """ """ """ """ """ """ """ """ """ """
        segmentator = LaneNet(weights_path=WEIGHTS_PATH)
        cluster = DbscanBased()
        fitter = LeastsBased()
        trans_mat33_, trans_mat33_r_ = himg.setPerspectiveTransform(self.cfg)
        """ """

        """ """ """ """ """ """ """ """ """ """ """
        PREPARE GRAPH I/O TO VARIABLE
        """ """ """ """ """ """ """ """ """ """ """
        # Define Input and Ouput tensors
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = graph.get_tensor_by_name('detection_scores:0')
        detection_classes = graph.get_tensor_by_name('detection_classes:0')
        num_detections = graph.get_tensor_by_name('num_detections:0')
        """ """

        """ """ """ """ """ """ """ """ """ """ """
        START WORKER THREAD
        """ """ """ """ """ """ """ """ """ """ """
        gpu_opts_object = [detection_boxes, detection_scores, detection_classes, num_detections]
        gpu_tag_object_distant = 'GPU_OBJECT'
        gpu_worker_object_distant = SessionWorker(gpu_tag_object_distant, graph, config)

        gpu_tag_lane = 'GPU_LANE'
        gpu_worker_lane = SessionWorker(gpu_tag_lane, graph, config, sess=segmentator.sess)
        gpu_opts_lane = [segmentator.binary_seg_ret]
        """ """

        """ """ """ """ """ """ """ """ """ """ """
        START IMAGE PIPE
        """ """ """ """ """ """ """ """ """ """ """
        q_in = Queue.Queue()
        q_out = Queue.Queue()
        q_out_bv = Queue.Queue()
        start_receiver(MPVariable.det_in_con, q_in, MPVariable.det_drop_frames)
        start_sender(MPVariable.det_out_con, q_out)
        start_sender(MPVariable.det_out_bv_con, q_out_bv)
        """ """

        """ """ """ """ """ """ """ """ """ """ """
        PERCEPTION LOOP
        """ """ """ """ """ """ """ """ """ """ """
        in_frame = None
        object_in_frame = None
        object_result_boxes = None
        object_result_scores = None
        object_result_classes = None
        object_result_num = None
        image_binarized_transformed = None
        lane_l_dict_nearest = None
        lane_r_dict_nearest = None
        lane_m_dict = None

        last_work_time_object = time.time()
        last_work_time_lane = time.time()

        try:
            while True:
                """
                Update in_frame
                """
                if gpu_worker_object_distant.is_sess_empty() \
                    or gpu_worker_lane.is_sess_empty():
                    # Check new queue
                    if q_in.empty():
                        time.sleep(0.001)
                        continue
                    # Get new image
                    q = q_in.get(block=False)
                    if q is None:
                        q_in.task_done()
                        break
                    in_frame = q['image']
                    q_in.task_done()

                """
                Add object tf worker task
                """
                if gpu_worker_object_distant.is_sess_empty():
                    # object detection input
                    input_frame_distant = in_frame[in_frame.shape[0]//2-object_clip_window_distant[0]//2 : in_frame.shape[0]//2+object_clip_window_distant[0]//2, 
                                                   in_frame.shape[1]//2-object_clip_window_distant[1]//2 : in_frame.shape[1]//2+object_clip_window_distant[1]//2].copy()
                    image_expanded_distant = np.expand_dims(cv2.cvtColor(input_frame_distant, cv2.COLOR_BGR2RGB), axis=0)
                    object_in_frame = input_frame_distant

                    # put new object distant queue
                    gpu_feeds_object_distant = {image_tensor: image_expanded_distant}
                    gpu_worker_object_distant.put_sess_queue(gpu_opts_object, gpu_feeds_object_distant, {})

                """
                Add lane tf worker task
                """
                if gpu_worker_lane.is_sess_empty():
                    # prepare lane detection input
                    image_resize = cv2.resize(in_frame, (512, 256), interpolation=cv2.INTER_LINEAR)
                    image_segmentator_input = segmentator.pre_binarize(image_resize)
                    image_segmentator_input_expanded = np.expand_dims(image_segmentator_input, axis=0)

                    # put new lane queue
                    gpu_feeds_lane = {segmentator.input_tensor: image_segmentator_input_expanded}
                    gpu_worker_lane.put_sess_queue(gpu_opts_lane, gpu_feeds_lane, {})

                """
                Object detection
                """
                result_queue_gpu_object_distant = gpu_worker_object_distant.get_result_queue()
                if result_queue_gpu_object_distant is not None:
                    object_result_boxes = result_queue_gpu_object_distant['results'][0]
                    object_result_scores = result_queue_gpu_object_distant['results'][1]
                    object_result_classes = result_queue_gpu_object_distant['results'][2]
                    object_result_num = result_queue_gpu_object_distant['results'][3]
                    object_result_boxes = np.squeeze(object_result_boxes)
                    object_result_scores = np.squeeze(object_result_scores)
                    object_result_classes = np.squeeze(object_result_classes)
                    # calc fps
                    MPVariable.perception_object_fps.value = int(1.0 / (time.time() - last_work_time_object))
                    last_work_time_object = time.time()

                """
                Lane detection
                """
                result_queue_gpu_lane = gpu_worker_lane.get_result_queue()
                if result_queue_gpu_lane is not None:
                    # binarize
                    image_binarized, extras_lane = result_queue_gpu_lane['results'][0], result_queue_gpu_lane['extras']
                    image_binarized = image_binarized[0].astype(np.uint8)
                    # transfrom
                    image_binarized_transformed = himg.transformImage(image_binarized, trans_mat33_)
                    # clustering
                    dbscan_input_array, dbscan_label, dbscan_label_n = cluster.clusterize(image_binarized_transformed)
                    # fitting
                    vehicle_cordinated_array, \
                    lane_l_dict_nearest, lane_r_dict_nearest, lane_m_dict = fitter.fitting(dbscan_input_array, dbscan_label, dbscan_label_n, \
                                                                                           cluster.dbscan_input_image.shape[1], \
                                                                                           cluster.dbscan_input_image.shape[0])
                    # calc fps
                    MPVariable.perception_lane_fps.value = int(1.0 / (time.time() - last_work_time_lane))
                    last_work_time_lane = time.time()

                    """
                    Update gloval value
                    """
                    # lane midle
                    MPVariable.lane_m_leasts_status.value = lane_m_dict['status']
                    MPVariable.lane_m_leasts_abc_lpf_a.value = lane_m_dict['leasts_abc_lpf'][0]
                    MPVariable.lane_m_leasts_abc_lpf_b.value = lane_m_dict['leasts_abc_lpf'][1]
                    MPVariable.lane_m_leasts_abc_lpf_c.value = lane_m_dict['leasts_abc_lpf'][2]
                    MPVariable.lane_m_width_lpf.value = lane_m_dict['lane_width_lpf']

                    # lane left
                    if len(lane_l_dict_nearest):
                        MPVariable.lane_l_leasts_status.value = True
                        MPVariable.lane_l_leasts_abc_lpf_a.value = lane_l_dict_nearest['leasts_abc_lpf'][0]
                        MPVariable.lane_l_leasts_abc_lpf_b.value = lane_l_dict_nearest['leasts_abc_lpf'][1]
                        MPVariable.lane_l_leasts_abc_lpf_c.value = lane_l_dict_nearest['leasts_abc_lpf'][2]
                        MPVariable.lane_l_leasts_z_max.value = lane_l_dict_nearest['z_max']
                    else:
                        MPVariable.lane_l_leasts_status.value = False

                    # lane right
                    if len(lane_r_dict_nearest):
                        MPVariable.lane_r_leasts_status.value = True
                        MPVariable.lane_r_leasts_abc_lpf_a.value = lane_r_dict_nearest['leasts_abc_lpf'][0]
                        MPVariable.lane_r_leasts_abc_lpf_b.value = lane_r_dict_nearest['leasts_abc_lpf'][1]
                        MPVariable.lane_r_leasts_abc_lpf_c.value = lane_r_dict_nearest['leasts_abc_lpf'][2]
                        MPVariable.lane_r_leasts_z_max.value = lane_r_dict_nearest['z_max']
                    else:
                        MPVariable.lane_r_leasts_status.value = False

                """
                Update output queue
                """
                if result_queue_gpu_object_distant is not None \
                    or result_queue_gpu_lane is not None:
                    q_out.put({'image': in_frame, \
                               'object_in_frame': object_in_frame, \
                               'object_result_boxes': object_result_boxes, \
                               'object_result_scores': object_result_scores, \
                               'object_result_classes': object_result_classes, \
                               'object_result_num': object_result_num, \
                               'lane_result_image_binarized_transformed': image_binarized_transformed, \
                               'lane_result_lane_l_dict_nearest': lane_l_dict_nearest, \
                               'lane_result_lane_r_dict_nearest': lane_r_dict_nearest, \
                               'lane_result_lane_m_dict': lane_m_dict})
                    q_out_bv.put({'dbscan_label': dbscan_label, \
                                  'dbscan_label_n': dbscan_label_n, \
                                  'vehicle_cordinated_array': vehicle_cordinated_array})

        except KeyboardInterrupt:
            pass
        except:
            import traceback
            traceback.print_exc()
        finally:
            q_out.put(None)
            q_out_bv.put(None)

        return
