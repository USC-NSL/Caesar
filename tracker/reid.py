import numpy as np  
from collections import defaultdict, OrderedDict
import logging 
from tracker.topo_matcher import TopoMatcher
from tracker.feature_extractor import feature_distance


# change these values for your testing scenes 
MIN_TUBE_DURATION = 4  # tube with enough clips should be considered matching
END_FRAME_NUM_THRES = 5  # tube that ended long enough can be considered 
MAX_TUBE_INFO_SIZE = 80  # max number of tubes cached for each camera
FEATURE_MATCHING_THRES = 0.4  # reid thres for feature matching 

class REID:
    def __init__(self, topo_path, img_shape):
        ''' For re-id people across cameras 
        '''
        # map one tube id to another if they belong to same person
        # (cam, tid) : (cam, tid) 
        self.id_mapping = {}  

        # Saves the tube's features and their last appearance frame 
        # (cam) : {'tid': {'last_frame_id':int, 
        #                   'last_box_pos' box, 
        #                   'feature': feature list,
        #                   'tube_len': number of frames}}
        self.tube_info = defaultdict(lambda: OrderedDict())   
        self.topo = TopoMatcher(topo_path, img_shape)

    def find_best_match(self, matched, frame_id):
        ''' For all matched tubes, select the best one 
            Strategy: order by feature dist then by time 
            The closer the better 
        Args:
        - matched: {(cid, tid): feature_dist} 
        ''' 
        dist_order = [[k, int(dist * 10.)] for k, dist in matched.items()]
        dist_order.sort(key=lambda x: x[1])
        min_time_gap = 9999
        res_cid, res_tid = '', 0

        for k, dist in dist_order:
            if dist > dist_order[0][1]:  # skip if the feature dist is not smallest
                break 
            cid, tid = k
            time_gap = frame_id - self.tube_info[cid][tid]['last_frame_id']
            if time_gap < min_time_gap:
                res_cid, res_tid = cid, tid
                min_time_gap = time_gap

        return res_cid, res_tid

    def generate_tube_frames(self, box_list, frame_list):
        ''' Output a list of cropped imgs for the tube 
        '''
        res = []
        frame_index = 0
        for fid, box in box_list:
            while frame_index < len(frame_list) and frame_list[frame_index][0] < fid:
                frame_index += 1
                continue 
            if frame_index >= len(frame_list):
                break 
            frame = frame_list[frame_index][1]
            res.append(frame[box[1]: box[3], box[0]: box[2]])
            
        return res

    def get_id(self, cid, tid, frame_id, box, feature):
        '''
        Assign a new tube id if for current tube_id: see if the tube could come 
        from another cam, if so compare its feature with all previous finised tubes' 
        fetures in other cams in reversed order

        Args:
        - cam_id, tube_id: the tracking result from tracker
        - frame_id: the current frame id 
        - box: the current box 
        - feature: the reid feature of the box   
        
        # box_list: a list of tuples (frame_id, box) of the tube
        # frame_list: a list of tuples (frame_id, frame) of the camera

        Return: 
        - new cam_id, new track_id, whether it's just re-ided
        '''
        # skip no feature tubes 
        if not len(feature):
            return cid, tid, False 

        # skip too short tubes 
        if tid not in self.tube_info[cid]:
            self.tube_info[cid][tid] = {
                                'tube_len': 0,
                                'first_box_pos': box,
                                'features': [feature],
                            }
        self.tube_info[cid][tid]['tube_len'] += 1

        # calculate the mean of the first few features for the tube 
        if self.tube_info[cid][tid]['tube_len'] < MIN_TUBE_DURATION:
            self.tube_info[cid][tid]['features'].append(feature)
            return cid, tid, False 
        elif self.tube_info[cid][tid]['tube_len'] == MIN_TUBE_DURATION:
            self.tube_info[cid][tid]['feature'] = np.mean(self.tube_info[cid][tid]['features'], axis=0)
            self.log('%s-%d confirmed' % (cid, tid))

        # update the last frame_id and box position 
        self.tube_info[cid][tid]['last_frame_id'] = frame_id
        self.tube_info[cid][tid]['last_box_pos'] = box
        
        # return if the tube is previously re-ided
        if (cid, tid) in self.id_mapping:
            cid, tid = self.id_mapping[cid, tid]
            if tid in self.tube_info[cid]:
                tmp = self.tube_info[cid][tid]
                del self.tube_info[cid][tid]
                self.tube_info[cid][tid] = tmp
            return cid, tid, False
        
        # Try to match current tube with other tubes 
        # matching dict: {(cid, tid): feature_dist} 
        matched = {}  
        for c in self.tube_info:
            # skip if two cameeras are not connected or the same 
            if c == cid or not self.topo.connected_camera(cid, c):
                continue 

            for t in self.tube_info[c]:
                tb = self.tube_info[c][t]
                if 'feature' not in tb:
                    continue 

                prev_fid, prev_feature = tb['last_frame_id'], tb['feature']
                prev_pos = tb['last_box_pos']

                tube_feature = self.tube_info[cid][tid]['feature']
                first_box_pos = self.tube_info[cid][tid]['first_box_pos']

                # skip if the enter/exit region doesn't match the topo file 
                if not self.topo.can_be_matched(cam1=cid, box1=first_box_pos,
                                                cam2=c, box2=prev_pos):
                    continue 

                '''
                self.log('try cur:%s-%d(%d>%s) to prev:%s-%d(%d>%s)' % (
                            cid, tid, frame_id, str(first_box_pos),
                            c, t, prev_fid, str(prev_pos)
                        ))
                '''

                # skip if the previous tube is still live 
                if (frame_id - END_FRAME_NUM_THRES) < prev_fid:
                    continue 

                fd = feature_distance(tube_feature, prev_feature) 
                if fd < FEATURE_MATCHING_THRES:
                    matched[c, t] = fd

        if matched:
            self.log('match (%s:%d) to %s' % (cid, tid, 
                                            str({k: '%0.2f' % matched[k] for k in matched})))
            cid_new, tid_new = self.find_best_match(matched, frame_id)
            self.id_mapping[cid, tid] = (cid_new, tid_new)
            cid, tid = cid_new, tid_new

        # remove too old tubes 
        if len(self.tube_info[cid]) > MAX_TUBE_INFO_SIZE:
            k = list(self.tube_info[cid].keys())[-1]
            del self.tube_info[cid][k]

        return cid, tid, len(matched) > 0

    def update(self, pkt):
        '''
        Main function of REID 

        Args:
        - pkt: the DataPkt from FExtractor (in network/data_packet.py)

        Return:
        - pkt: the DataPkt with meta updated, sent to the next hop
        '''
        for m in pkt.meta:
            # Skip untrackable objects 
            if 'id' not in m or 'feature' not in m:
                continue 

            tube_id, box, feature = m['id'], m['box'], m['feature']
            cid, tid, reided = self.get_id(pkt.cam_id, tube_id, pkt.frame_id, box, feature)
            if reided:
                self.log('!! %d > [%s-%s] [%s-%s]' % (pkt.frame_id, pkt.cam_id, tube_id, cid, tid))
                m['reid'] = (cid, tid)

        return pkt

    def log(self, s):
        logging.debug('[REID] %s' % s)
