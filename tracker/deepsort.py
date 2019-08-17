import numpy as np
from os.path import join
import os
import sys
from time import time
import logging 
from tracker.deep_sort.detection import Detection
from tracker.deep_sort.tracker import Tracker
from tracker.deep_sort import nn_matching


SAME_TUBE_IOU_MIN = 0.3
def get_iou(a, b, epsilon=1e-5):
    """
    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou


class DeepSort:
    def __init__(self, track_labels=[], attach_labels=[]):
        self.trackers = {}
        for i in track_labels:
            metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
            self.trackers[i] = Tracker(metric, max_iou_distance=0.7,
                                        max_age=100, n_init=4)

        self.track_labels = track_labels
        self.attach_labels = attach_labels
        self.log('init')

    def find_box_feature(self, box, all_meta):
        """
        Given a tracked box and all src meta (with 'box', 'label', and 'feature'),
        return the feature that matched to this box. If none matched, return empty
        """
        max_iou = SAME_TUBE_IOU_MIN
        res = []
        for m in all_meta:
            iou = get_iou(box, m['box'])
            if iou > max_iou and 'feature' in m:
                res = m['feature']
                max_iou = iou
        return res
        
    def update(self, pkt):
        '''
        Args:
        - pkt: the DataPkt from FExtractor (in network/data_packet.py)

        Return:
        - pkt: the DataPkt with track id added, sent to the next hop
        '''
        track_boxes = {i:[] for i in self.track_labels}
        track_scores = {i:[] for i in self.track_labels}
        track_features = {i:[] for i in self.track_labels}
        out_meta = []

        for m in pkt.meta:
            box = m['box']
            label = m['label']
            score = m['score']
            feature = m['feature'] if 'feature' in m else []
            if label in self.track_labels:
                track_boxes[label].append([box[0], box[1],
                                            box[2]-box[0], box[3]-box[1]])
                track_scores[label].append(score)
                track_features[label].append(feature)
            elif label in self.attach_labels:
                out_meta.append(m)

        for label in track_boxes:
            detection_list = [Detection(track_boxes[label][i],
                                        track_scores[label][i],
                                        track_features[label][i])
                                    for i in range(len(track_boxes[label]))]

            self.trackers[label].predict()
            self.trackers[label].update(detection_list)

            for tk in self.trackers[label].tracks:
                if not tk.is_confirmed() or tk.time_since_update > 1:
                    continue
                left, top, width, height = tk.to_tlwh()
                box = [int(left), int(top), int(left + width), int(top + height)]
                out_meta.append({'box': box,
                                'id': tk.track_id,
                                'feature': self.find_box_feature(box, pkt.meta),
                                'label': label})

        # self.log(str([(m['id'], len(m['feature'])) for m in out_meta]))
        pkt.meta = out_meta
        return pkt


    def log(self, s):
        logging.debug('[DeepSort] %s' % s)
