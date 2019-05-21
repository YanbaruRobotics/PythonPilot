#!/usr/bin/python3
# -*- coding: utf-8 -*-

from object_detection.tf_utils import label_map_util


class LoadLabelMap():
    def __init__(self):
        return
    
    def load_label_map(self, cfg):
        """
        LOAD LABEL MAP
        """
        print('Loading label map')
        LABEL_PATH           = cfg['label_path']
        NUM_CLASSES          = cfg['num_classes']
        try:
            label_map = label_map_util.load_labelmap(LABEL_PATH)
            categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
            category_index = label_map_util.create_category_index(categories)
        except:
            import traceback
            traceback.print_exc()
        return category_index

