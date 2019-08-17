import sys
import cv2 
import logging
import numpy as np
from time import time, sleep
from collections import defaultdict, OrderedDict
from server.action_graph import Act
from server.action_spatial import overlap
from network.data_packet import DataPkt
from network.utils import Q


CONTEXT_BOX_RATIO = 1.3
CROP_IMG_SIZE = (400, 400)

class ServerPkt:
    def __init__(self, cam_id, pkts, tubes, reid={}):
        ''' Pkt passed from TM to Spatial, NN, and Complex Act
            So they can use the frame_id, cam_id, and label to fetch the tube imgs
            and the original imgs

        Input:
        - cam_id: the camera id
        - pks: a list of cached packets (DataPkt)
        - tubes: a list of tubes in the time period (Tube)
        - reid: a dict of reid mapping (cur_tid -> [prev_cid, prev_tid])
        '''
        self.cam_id = cam_id
        self.pkts = pkts
        self.tubes = tubes
        self.reid = reid

        self.actions = []     # a list of actions (server/action_graph.py - Act)
        for tube in tubes:
            for obj in tube.overlap_objs:  # extract the attach object as action
                self.actions.append(Act(act_name="with_{}".format(obj),
                                        class1=tube.label,
                                        tube1=tube.tube_id,
                                        frame_id=tube.tube_clips[0].frame_id))

    def get_first_frame_id(self):
        """ return the frame_id of the first frame """
        return self.pkts[0].frame_id

    def get_action_metas(self):
        """ Return all actions in meta format """
        return [a.to_meta() for a in self.actions]

    def get_action_info(self):
        """ Return all action's info to console """
        return [a.to_log() for a in self.actions]

    def to_data_pkts(self, mode='all'):
        """
        Return a generator of data packets, with differet meta output modes

        @param: mode str:
                    "tube" only show obj bbox,
                    "act" only show acts,
                    "all" show both
        """
        act_meta = [] if mode == 'tube' else self.get_action_metas()
        if mode == 'all':
            self.pkts[0].meta += act_meta
        elif mode == 'act':
            self.pkts[0].meta = act_meta
            for i in range(1, len(self.pkts)):
                self.pkts[i].meta = []

        for p in self.pkts:
            yield p


def generate_clip_image_roi(box, whole_frame, context_box_ratio=CONTEXT_BOX_RATIO, dst_img_size=CROP_IMG_SIZE):
    """
    Return image and the roi for action detection tube image (each frame)

    Params: 
    - box: absolute x0, y0, x1, y1 in the whole frame 
    - whole_frame: the whole frame 
    - context_box_ratio: the context square's edge length L = (w + h) * context_box_ratio
    - dst_img_size: resize the context square to this shape 

    Return:
    img, roi
    """
    H, W, _ = whole_frame.shape
    box_center = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]
    box_wid = (box[2] - box[0])
    box_hei = (box[3] - box[1])
    edge = int(min((box_wid + box_hei) * context_box_ratio, H))
    h_edge = edge // 2     # half edge size 

    left_bound = max(0, box_center[0] - h_edge + 1)
    bottom_bound = max(0, box_center[1] - h_edge + 1)
    right_bound = min(W - 1, box_center[0] + h_edge - 1)
    top_bound = min(H - 1, box_center[1] + h_edge - 1)

    frame_crop = whole_frame[bottom_bound:top_bound, left_bound:right_bound]
    left_paste_pos = h_edge - (box_center[0] - left_bound)
    bottom_paste_pos = h_edge - (box_center[1] - bottom_bound)

    img_crop = np.zeros((edge, edge, 3), np.uint8)
    img_crop[bottom_paste_pos:bottom_paste_pos+(top_bound-bottom_bound),
            left_paste_pos:left_paste_pos+(right_bound-left_bound)] = frame_crop

    img = cv2.resize(img_crop, dst_img_size)
    roi = [(h_edge - box_wid/2) / edge, (h_edge - box_hei/2) / edge, 
            (h_edge + box_wid/2) / edge, (h_edge + box_hei/2) / edge]
    
    return img, roi


class TubeClip:
    def __init__(self, box, frame_id, whole_frame):
        """
        Params: 
        - box: absolute x0, y0, x1, y1 in the whole frame 
        - frame_id: ..
        - whole_frame: the whole frame 
        """
        self.box = box
        self.frame_id = frame_id
        self.img, self.roi = generate_clip_image_roi(box, whole_frame)

    
class Tube:
    def __init__(self, label, tube_id):
        ''' 
        Args:
        - label: the class of the tube (people, car, etc.)
        - tube_id: the tracking id 
        - tube_clips: a list of TubeClip that contains box, tube img, roi, frame id 
        - overlap_objs: tags of bag, bicycle... only for person
        '''
        self.label = label
        self.tube_id = tube_id
        self.tube_clips = []
        self.overlap_objs = set()
    

    def add_tube_clip(self, box, frame_id, img):
        """
        Add one tube clip to the tube 
        """
        self.tube_clips.append(TubeClip(box, frame_id, img))
        

class PktCache:
    def __init__(self, track_list, overlap_list, max_tube_size, min_tube_size):
        ''' Cache the pkts of a camera, other modules can query for original images
        '''
        self.pkts = []                  # a list of pkts for this camera

        self.track_list = track_list  # list of obj labels that will be tracked
        self.overlap_list = overlap_list  # list of objs that will be attachement
        self.max_tube_size = max_tube_size  # output when receive max_size new pkts
        self.min_tube_size = min_tube_size  # min frame number of a valid tube
        self.reid = {}

        self.log('init')

    def is_full(self):
        """ return true if cache is full """
        return len(self.pkts) >= self.max_tube_size

    def get_all_valid_tubes(self):
        ''' Return a dict {label: [tube_id]} of tubes that long enough
        '''
        valid_tubes = defaultdict(int)

        for pkt in self.pkts:
            for obj in pkt.meta:
                if 'id' in obj and obj['label'] in self.track_list:
                    valid_tubes[obj['label'], obj['id']] += 1

        res = set()
        for label, tid in valid_tubes.keys():
            if valid_tubes[label, tid] >= self.min_tube_size:
                res.add((label, tid))
        return res

    def get_overlap_obj_list(self):
        ''' Return a list of overlap obj box position: [{label: [box_pos]}]
        '''
        res = []
        for pkt in self.pkts:
            cur = {}
            for obj in pkt.meta:
                label = obj['label']
                if label in self.overlap_list:
                    if not label in cur:
                        cur[label] = []
                    cur[label].append(obj['box'])
            res.append(cur)

        return res

    def generate_tubes(self):
        '''
        Return a lis of tubes for all valid tubes in the cacahed packets
        '''
        valid_track_ids = self.get_all_valid_tubes()
        overlap_objs = self.get_overlap_obj_list()

        res = {}
        for label, tid in valid_track_ids:
            res[label, tid] = Tube(label, tid)

        for i, pkt in enumerate(self.pkts):
            for obj in pkt.meta:
                if 'id' not in obj:   # skip item that not being tracked
                    continue

                label, tid, box = obj['label'], obj['id'], obj['box']
                if (label, tid) not in res:   # skip tubes that are too short
                    continue

                if 'reid' in obj:     # mark original tid and cid if tube just reided
                    self.reid[tid] = obj['reid']

                tube = res[label, tid]
                tube.add_tube_clip(box, pkt.frame_id, pkt.img)

                # find overlapped obj for person tubes
                if label == 'person':
                    for obj_label in overlap_objs[i]:
                        for obj_box in overlap_objs[i][obj_label]:
                            if overlap(box, obj_box):
                                tube.overlap_objs.add(obj_label)
                                break

        return [tube for _, tube in res.items()]

    def log(self, s):
        logging.debug('[PktCache] %s' % s)


MAX_TUBE_SIZE_DEFAULT = 16
MIN_TUBE_SIZE_DEFAULT = 8

class ServerPktManager:
    def __init__(self, in_queue, out_queue, track_list, overlap_list,
                    max_tube_size=MAX_TUBE_SIZE_DEFAULT,
                    min_tube_size=MIN_TUBE_SIZE_DEFAULT):
        ''' Batch Datapkt (frames and boxes) into ServerPkt (frame_batch and tubes)
            Also extract attachment object for person tubes, filter too short tubes,
            and re-identify person tubes. Main function: run

        Args:
        - in_queue: queue that contains input data
        - out_queue: queue that contains output data
        - track_list: a list of trackable labels
        - overlap_list: a list of overlappable labels for human attachment
        - max_tube_size: max # of frames in a tube
        - min_tube_size: min # of frames in a tube
        '''

        # temporary cache the packets
        self.caches = defaultdict(lambda: PktCache(track_list=track_list,
                                                overlap_list=overlap_list,
                                                max_tube_size=max_tube_size,
                                                min_tube_size=min_tube_size))
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.log('init')

    def run(self):
        '''
        Input: read from in_queue, input is DataPkt (network.data_packet)

        Output: write to outqueue, output is ServerPkt (server.server_packet)
        '''
        while True:
            pkt = self.in_queue.read()
            if pkt is None:
                sleep(0.01)
                continue

            cam_id = pkt.cam_id
            cache = self.caches[cam_id]
            cache.pkts.append(pkt)

            if not cache.is_full():  # continue if cache not full
                continue

            # init the server pkt with frames and frame ids
            output_pkt = ServerPkt(cam_id=cam_id,
                                    pkts=cache.pkts,
                                    tubes=cache.generate_tubes(),
                                    reid=cache.reid)

            self.out_queue.write(output_pkt)
            cache.pkts = []
            cache.reid = {}

    def log(self, s):
        logging.debug('[ServerPktManager] %s' % s)


# debug
if __name__ == '__main__':
    print('done')
