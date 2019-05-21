#!/usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
import multiprocessing
import numpy as np
import queue as Queue
import time

from perception.object_detection.load_label_map import LoadLabelMap
from perception.lane_fitting.leasts_based import LeastsBased
from perception.perception_common import handling_image as himg

from common.pilot_gloval_variable import MPVariable
from common.camera_img_pipe import start_receiver

from object_detection.tf_utils import visualization_utils_cv2 as vis_util


def visualize_text(image, cfg, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.8, fontThickness=1, draw_back_box=True):
    textMode = cfg['vis_textmode']
    display_str = []
    max_text_width = 0
    max_text_height = 0

    display_str.append("camera_frame_counter: {}".format(MPVariable.camera_frame_counter.value))
    display_str.append("perception_object_fps: {:.1f}".format(MPVariable.perception_object_fps.value))
    display_str.append("perception_lane_fps: {:.1f}".format(MPVariable.perception_lane_fps.value))
    display_str.append("")
    display_str.append("lane_s [-]:   {}".format(MPVariable.lane_m_leasts_status.value))
    display_str.append("lane_m_w [m]:   {:.3f}".format(MPVariable.lane_m_width_lpf.value))
    display_str.append("lane_m_a [1/m]: {:.6f}".format(MPVariable.lane_m_leasts_abc_lpf_a.value))
    display_str.append("lane_m_b [rad]: {:.6f}".format(MPVariable.lane_m_leasts_abc_lpf_b.value))
    display_str.append("lane_m_c [m]:   {:.3f}".format(MPVariable.lane_m_leasts_abc_lpf_c.value))
    display_str.append("lane_l_c [m]:   {:.3f}".format(MPVariable.lane_l_leasts_abc_lpf_c.value))
    display_str.append("lane_r_c [m]:   {:.3f}".format(MPVariable.lane_r_leasts_abc_lpf_c.value))
    display_str.append("")
    display_str.append("counter_servo_unit [-]:  {:6}".format(MPVariable.can_rx_counter_servo_unit.value))
    display_str.append("time_ms_diff [ms]:       {:6}".format(MPVariable.can_rx_time_us_diff.value))
    display_str.append("buttons [-]: {} {} {}".format(MPVariable.can_rx_button_y.value, \
                                                      MPVariable.can_rx_button_g.value, \
                                                      MPVariable.can_rx_button_r.value))
    display_str.append("actual_angle [deg]: {:.3f}".format(MPVariable.can_rx_actual_angle.value))
    display_str.append("")
    display_str.append("counter_camera_unit [-]: {:6}".format(MPVariable.can_tx_counter_camera_unit.value))
    display_str.append("servo_on_flag [-] : {}".format(MPVariable.can_tx_servo_on_flag.value))
    display_str.append("target_angle [deg]: {:.3f}".format(MPVariable.can_tx_target_angle.value))
    display_str.append("")
    display_str.append("can_error_count_tx [-]:  {}".format(MPVariable.can_error_count_tx.value))
    display_str.append("")
    display_str.append("obd_temp [deg.c.]:  {}".format(MPVariable.obd_temp_degrees_celsius.value))
    display_str.append("obd_engine_speed [r/min]:  {}".format(MPVariable.obd_engine_speed_rpm.value))
    display_str.append("obd_vehicle_speed [km/h]:  {}".format(MPVariable.obd_vehicle_speed_kmph.value))

    """ DRAW BLACK BOX AND TEXT """
    [(text_width, text_height), baseLine] = cv2.getTextSize(text=display_str[0], fontFace=fontFace, fontScale=fontScale, thickness=fontThickness)
    x_left = int(baseLine) + 850
    y_top = int(baseLine) + 10
    for i in range(len(display_str)):
        [(text_width, text_height), baseLine] = cv2.getTextSize(text=display_str[i], fontFace=fontFace, fontScale=fontScale, thickness=fontThickness)
        if max_text_width < text_width:
            max_text_width = text_width
        if max_text_height < text_height:
            max_text_height = text_height

    # """ DRAW BLACK BOX """
    if draw_back_box:
        image_back = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        cv2.rectangle(image_back, \
                      (x_left - 2, int(y_top)), \
                      (int(x_left + max_text_width + 2), int(y_top + len(display_str)*max_text_height*1.6+baseLine)), \
                      color=(255, 255, 255), \
                      thickness=-1)
        if textMode == 'day':
            beta = -0.6
        else:
            beta = 0.2
        image = cv2.addWeighted(image, 1.0, image_back, beta, 0.0)

    """ DRAW FPS, TEXT """
    for i in range(len(display_str)):
        cv2.putText(image, \
                    display_str[i], \
                    org=(x_left, y_top + int(max_text_height*1.6 + (max_text_height*1.6 * i))), \
                    fontFace=fontFace, fontScale=fontScale, thickness=fontThickness, color=(0, 255, 0))

    cv2.putText(image, \
        "z: {:.3f}".format(MPVariable.pp_target_z.value), \
        org=(x_left-200, image.shape[0]-180), \
        fontFace=fontFace, fontScale=1.6, thickness=fontThickness, color=(0, 255, 0))
    cv2.putText(image, \
        "c: {:.3f}".format(MPVariable.lane_m_leasts_abc_lpf_c.value), \
        org=(x_left-200, image.shape[0]-130), \
        fontFace=fontFace, fontScale=1.6, thickness=fontThickness, color=(0, 255, 0))

    cv2.putText(image, \
        "actual: {:.3f}".format(MPVariable.can_rx_actual_angle.value), \
        org=(x_left, image.shape[0]-180), \
        fontFace=fontFace, fontScale=1.6, thickness=fontThickness, color=(0, 255, 0))
    cv2.putText(image, \
        "target: {:.3f}".format(MPVariable.can_tx_target_angle.value), \
        org=(x_left, image.shape[0]-130), \
        fontFace=fontFace, fontScale=1.6, thickness=fontThickness, color=(0, 255, 0))

    return image


class MPVisualization():
    def __init__(self, cfg):
        self.__m = multiprocessing.Process(target=self.execution, args=(cfg,))
        self.__m.start()
        return

    def end(self):
        """
        End process
        """
        self.__m.join()
        print('Finish MPVisualization')
        return

    def execution(self, cfg):
        object_clip_window_distant = [256, 512]

        trans_mat33_, trans_mat33_r_ = himg.setPerspectiveTransform(cfg)
        fitter = LeastsBased()

        llm = LoadLabelMap()
        category_index = llm.load_label_map(cfg)

        q_in = Queue.Queue()
        start_receiver(MPVariable.vis_in_con, q_in, MPVariable.vis_drop_frames)

        try:
            while True:
                if q_in.empty():
                    time.sleep(0.001)
                    continue

                q = q_in.get(block=False)
                if q is None:
                    q_in.task_done()
                    break

                in_frame = q['image']

                # prepare object boxes image
                if q['object_result_num'] is None:
                    continue
                image_boxes = q['object_in_frame']
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_boxes,
                    q['object_result_boxes'],
                    q['object_result_scores'],
                    q['object_result_classes'],
                    category_index,
                    instance_masks=None,
                    use_normalized_coordinates=True,
                    line_thickness=1,
                    min_score_thresh=0.1,
                    max_boxes_to_draw=10)

                # overlay objects image
                out_frame = in_frame.copy()
                out_frame[out_frame.shape[0]//2-object_clip_window_distant[0]//2 : out_frame.shape[0]//2+object_clip_window_distant[0]//2, 
                          out_frame.shape[1]//2-object_clip_window_distant[1]//2 : out_frame.shape[1]//2+object_clip_window_distant[1]//2] = image_boxes

                # get image_binarized_transformed
                if q['lane_result_lane_m_dict'] is None:
                    continue
                image_binarized_transformed = q['lane_result_image_binarized_transformed']

                # draw lane
                steering_a = MPVariable.can_rx_actual_angle.value * 0.0001
                image_lanes = fitter.draw_fitting(image_binarized_transformed, \
                                                  trans_mat33_r_, \
                                                  q['lane_result_lane_l_dict_nearest'], \
                                                  q['lane_result_lane_r_dict_nearest'], \
                                                  q['lane_result_lane_m_dict'], \
                                                  steering_a)
                image_lanes = cv2.resize(image_lanes, (out_frame.shape[1], out_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                out_frame = cv2.addWeighted(out_frame, 1.0, image_lanes, 0.5, 0.0)


                # draw text
                out_frame = visualize_text(out_frame, cfg)

                cv2.imshow("Visualization Camera", out_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    q_in.task_done()
                    break

                q_in.task_done()

        except KeyboardInterrupt:
            pass
        except:
            import traceback
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()
        return
