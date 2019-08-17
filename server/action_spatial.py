import numpy as np
from time import time, sleep
from collections import defaultdict
from server.action_graph import Act
import logging

"""
Spatial actions utils
"""
def overlap(b1, b2):
    if b2[0] > b1[2] or b2[2] < b1[0]:
        return False 
    if b2[1] > b1[3] or b2[3] < b1[1]:
        return False 

    return True 


def box_dist(b1, b2):
    return (((b1[0] + b1[2]) / 2 - (b2[0] + b2[2]) / 2) ** 2 + \
            ((b1[1] + b1[3]) / 2 - (b2[1] + b2[3]) / 2) ** 2) ** 0.5 


def get_boxes_within_fid_range(tube, t1, t2):
    res = []
    for clip in tube.tube_clips:
        fid = clip.frame_id
        if fid > t2:
            break 
        if fid < t1:
            continue 
        res.append(clip.box)

    return res 


"""
Spatial actions definition
"""
# The two tubes should have at least this # of frames overlapped 
# so we can decide the relative movement of two tubes 
MIN_OVERLAP_FRAME_NUM = 8

# If the relative dist changes (relative to box size) more than 
# this value, We think there's an approaching/leaving event  
MOVEMENT_THRES_RATIO = 1.1

# The relative distance (compared w/ avg box wid) threshold 
CLOSE_MAX_RATIO = 1.8
NEAR_MAX_RATIO = 3

def tube_dist_relation(t1, t2):
    """
    Return if the two tubes are far, near, or close; Also return if 
    the two tubes (approach), (cross), or (leave) each other
    """
    start_fid = max(t1.tube_clips[0].frame_id, t2.tube_clips[0].frame_id)
    end_fid = min(t1.tube_clips[-1].frame_id, t2.tube_clips[-1].frame_id)
    if end_fid - start_fid < MIN_OVERLAP_FRAME_NUM:
        return []
    boxes1 = get_boxes_within_fid_range(t1, start_fid, end_fid)
    boxes2 = get_boxes_within_fid_range(t2, start_fid, end_fid) 
    b1_wid_avg = (boxes1[0][2] - boxes1[0][0]) / 2 +  (boxes1[-1][2] - boxes1[-1][0]) / 2

    start_dist = box_dist(boxes1[0], boxes2[0]) / b1_wid_avg
    mid_dist = box_dist(boxes1[len(boxes1)//2], boxes2[len(boxes2)//2]) / b1_wid_avg
    end_dist = box_dist(boxes1[-1], boxes2[-1]) / b1_wid_avg

    def _dist_relation(d):
        if d < CLOSE_MAX_RATIO:
            return 'close'
        if d < NEAR_MAX_RATIO:
            return 'near'
        return 'far'

    res = []
    if end_dist <= mid_dist and start_dist - end_dist > MOVEMENT_THRES_RATIO:
        res.append('approach')
    elif min(end_dist, start_dist) - mid_dist > MOVEMENT_THRES_RATIO:
        res.append('cross')
    elif start_dist <= mid_dist and end_dist - start_dist > MOVEMENT_THRES_RATIO:
        res.append('leave')

    res.append(_dist_relation(mid_dist))
    return res 


MOVING_SEG_SIZE = 10
MOVING_THRES_MOVE_RATIO = 0.4
MOVING_THRES_STOP_RATIO = 0.3
def moving_status(tube):
    """
    Return True if the tube is moving. We break the tube down into pieces that 
    each contains MOVING_SEG_SIZE frames. For each piece, we test if one moves 
    a MOVING_THRES_MOVE_RATIO distance or more (compared against the bbox size),
    if so, it is "move". Then we test with MOVING_THRES_STOP_RATIO, if less, it
    is "stop"

    arg: tube - includes label, tube_id, tube_clips, overlap_objs (not used)
    """
    clips = tube.tube_clips
    is_stop = True 
    for i in range(0, len(clips), MOVING_SEG_SIZE):
        first_clip = clips[i]
        last_clip = clips[min(i + MOVING_SEG_SIZE - 1, len(clips) - 1)]
        box_dimen = (first_clip.box[2] - first_clip.box[0]) / 2 + (last_clip.box[2] - last_clip.box[0]) / 2
        move_dist = box_dist(first_clip.box, last_clip.box)
        move_ratio = move_dist / box_dimen
        if move_ratio > MOVING_THRES_STOP_RATIO:
            is_stop = False 
            if move_ratio > MOVING_THRES_MOVE_RATIO:
                return 'move'

    return 'stop' if is_stop else ''


# If a tube has been inactive more than this gap, we can treat it has ended 
MAX_INACTIVE_FRAME_NUM = 120

class SpatialActDetector:
    def __init__(self, in_queue, out_queue):
        self.in_queue = in_queue
        self.out_queue = out_queue

        # A dict that record the last active frame_id of each tube: 
        # key-(cam_id, label, tube_id),     val-last_frame_id
        self.tube_reg = defaultdict(int)  

        self.log('init')

    def get_end_actions(self, cur_frame_id):
        ''' Add 'end' action for all old tubes and remove them from tube_reg

        Return: a list of Act (server.action_graph.py)
        '''
        remove_list = []
        for tube_key, last_active_fid in self.tube_reg.items():
            if last_active_fid < cur_frame_id - MAX_INACTIVE_FRAME_NUM:
                remove_list.append(tube_key)

        res = []
        for tube_key in remove_list:
            cam_id, label, tube_id = tube_key
            res.append(Act(act_name='end', class1=label, tube1=tube_id))
            del self.tube_reg[tube_key]

        return res

    def get_single_actions(self, cam_id, tube):
        ''' 
        Add 'start' action for the tube if it is new, and calculates the single
        actions for the current tube 

        Args:
        - cam_id: camera id of the current tube 
        - tube: includes label, tube_id, tube_clips, overlap_objs (not used)

        Return: a list of Act (server.action_graph.py)
        '''
        key = (cam_id, tube.label, tube.tube_id)
    
        res = []
        if key not in self.tube_reg:
            res.append(Act('start', class1=tube.label, tube1=tube.tube_id))         
        self.tube_reg[key] = tube.tube_clips[-1].frame_id
            
        act = moving_status(tube)
        if act:
            res.append(Act(act, class1=tube.label, tube1=tube.tube_id))

        return res

    def get_cross_actions(self, t1, t2):
        ''' Currently we handle (attach obj), (car-person), and (person person).

        Args: 
        - tube t1, t2: (label, tube_id, tube_clips, overlap_objs)

        Return: a list of Act (server.action_graph.py)
        '''
        if t1.label == t2.label == 'car':  # skip the both-car case
            return []
        if t1.label == 'car' and t2.label == 'person':
            t2, t1 = t1, t2

        res = []

        dist_relations = tube_dist_relation(t1, t2)
        for r in dist_relations:
            res.append(Act(r, t1.label, t1.tube_id, t2.label, t2.tube_id))
            res.append(Act(r, t2.label, t2.tube_id, t1.label, t1.tube_id))
        
        return res

    def run(self):
        ''' 
        Input: from in_queue, read ServerPkt (server.server_packet)
        Output: to out_queue, write ServerPkt (server.server_packet)
        '''
        logging.basicConfig(level=logging.DEBUG)
        while True:
            server_pkt = self.in_queue.read()
            if server_pkt is None:
                sleep(0.01)
                continue

            res = []
            tubes = server_pkt.tubes
            cam_id = server_pkt.cam_id
            for i in range(len(tubes)):
                res += self.get_single_actions(cam_id, tubes[i])
                for j in range(i + 1, len(tubes)):
                    res += self.get_cross_actions(tubes[i], tubes[j])

            # get "end" actions 
            res += self.get_end_actions(server_pkt.get_first_frame_id())

            server_pkt.actions += res
            self.out_queue.write(server_pkt)

    def log(self, s):
        logging.debug('[Spatial] %s' % s)
