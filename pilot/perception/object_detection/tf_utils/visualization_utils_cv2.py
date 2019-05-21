# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A set of functions that are used for visualization.

These functions often receive an image, perform some visualization on the image.
The functions do not return a value, instead they modify the image itself.

"""
import collections
import numpy as np
import cv2
from object_detection.tf_utils.color_map import STANDARD_COLORS, STANDARD_COLORS_ARRAY

_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10

def draw_bounding_box_on_image_cv(image,
                                  ymin,
                                  xmin,
                                  ymax,
                                  xmax,
                                  color=(0, 0, 255),
                                  thickness=4,
                                  display_str_list=(),
                                  use_normalized_coordinates=True):
  im_height, im_width = image.shape[:2]
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

  ####################
  # draw objectbox
  ####################
  points = np.array([[left, top], [left, bottom], [right, bottom], [right, top], [left, top]])
  cv2.polylines(image, np.int32([points]),
                isClosed=False, thickness=thickness, color=color, lineType=cv2.LINE_AA)

  ####################
  # calculate str width and height
  ####################
  fontFace = cv2.FONT_HERSHEY_SIMPLEX
  fontScale = 0.3
  fontThickness = 1
  display_str_heights = [cv2.getTextSize(text=ds, fontFace=fontFace, fontScale=fontScale, thickness=fontThickness)[0][1] for ds in display_str_list]
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height

  ####################
  # draw textbox and text
  ####################
  for display_str in display_str_list[::-1]:
    # 
    [(text_width, text_height), baseLine] = cv2.getTextSize(text=display_str, fontFace=fontFace, fontScale=fontScale, thickness=fontThickness)
    margin = np.ceil(0.05 * text_height)

    cv2.rectangle(image, (int(left), int(text_bottom - 3 * baseLine - text_height - 2 * margin)), (int(left + text_width), int(text_bottom - baseLine)), color=color, thickness=-1)
    cv2.putText(image, display_str, org=(int(left + margin), int(text_bottom - text_height - margin)), fontFace=fontFace, fontScale=fontScale, thickness=fontThickness, color=(0, 0, 0))
  
    text_bottom -= text_height - 2 * margin


def draw_bounding_box_on_image_array_cv(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color=(0, 255, 0),
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
  """Adds a bounding box to an image (numpy array).

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw_bounding_box_on_image_cv(image, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)
  return

def draw_mask_on_image_array_cv(image, mask, color=(0, 0, 255), alpha=0.4):
  """Draws mask on an image.
  Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    mask: a uint8 numpy array of shape (img_height, img_height) with
      values between either 0 or 1.
    color: color to draw the keypoints with. Default is red.
    alpha: transparency value between 0 and 1. (default: 0.4)
  Raises:
    ValueError: On incorrect data type for image or masks.
  """
  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if np.any(np.logical_and(mask != 1, mask != 0)):
    raise ValueError('`mask` elements should be in [0, 1]')
  if image.shape[:2] != mask.shape:
    raise ValueError('The image has spatial dimensions %s but the mask has '
                     'dimensions %s' % (image.shape[:2], mask.shape))
  mask = STANDARD_COLORS_ARRAY[mask]
  # cv2.addWeighted(cv_bgr_under, alpha_under, cv_bgr_upper, alpha_upper, gamma, output_image)
  cv2.addWeighted(mask, alpha, image, 1.0, 0, image)
  return


def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    scores,
    classes,
    category_index,
    instance_masks=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color=(0, 0, 0),
    skip_scores=False,
    skip_labels=False,
    skip_labels_array=[]):
  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or (scores[i] > min_score_thresh \
                            and category_index[classes[i]]['name']!="airplane"):
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[0][i]
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(int(100*scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
        box_to_display_str_map[box].append(display_str)
        if agnostic_mode:
          box_to_color_map[box] = (0, 140, 255) # 'DarkOrange'
        else:
          box_to_color_map[box] = STANDARD_COLORS[
            (int)(classes[i] % len(STANDARD_COLORS))]

  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    if xmax-xmin > 0.8:
      continue
    if instance_masks is not None:
      #print("==========")
      #np.set_printoptions(precision=5, threshold=np.inf, suppress=True)  # suppress scientific float notation
      #print(box_to_instance_masks_map[box])

      draw_mask_on_image_array_cv(
        image,
        box_to_instance_masks_map[box],
        color=color
      )
    draw_bounding_box_on_image_array_cv(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=line_thickness,
        display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)

  return image
