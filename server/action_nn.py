from collections import deque
from time import time, sleep
from threading import Thread 
import logging
import numpy as np
import server.acam.action_detector as act
from server.action_graph import Act
from web.visualizer import Visualizer


# will clean the tube from the cache if it's been inactive too long
# the newly refreshed tube's age is 0 
MAX_TUBE_AGE_IN_CACHE = 2

# after how many rounds we have to process a non-filled cache
MAX_NON_EMPTY_CACHE_ROUND = 2

# Get top k actions for each tube for the NN output 
TOP_K_ACT = 5

# NN Action List
NN_ACT_THRES = {
                    'use_phone':0.2, 
                    'carry':0.1, 
                    'use_computer':0.2, 
                    'talk':0.2, 
                    'sit':0.2,
                    'ride':0.18,
                }

class NNActDetector:
    def __init__(self, in_queue, out_queue, model_path, batch_size, tube_size, filter_queue):
        self.in_queue = in_queue
        self.out_queue = out_queue

        self.model_path = model_path
        self.detector_dict = {}

        # how many tubes are processed together with one NN run 
        self.batch_size = batch_size
        # the lenght of tube that will be feed to NN 
        self.tube_size = tube_size
        # this queue includes all the tube ids that we need to run NN on
        self.filter_queue = filter_queue
        # how many rounds that the cache is non empty and not processed
        self.non_empty_cache_round = 0
        # tubes that will be processed in current round
        self.cache = deque()
        # a dummy tube data to fill the batch for the NN 
        self.dummy_tube = self.get_dummy_tube()

    def get_dummy_tube(self):
        """ 
        Generate an empty tube data to fill the NN input 
        """
        dummy_img = np.zeros((400, 400, 3))
        dummy_roi = [0.25,0.25,0.75,0.75]
        return {
                'imgs': [dummy_img for _ in range(self.tube_size)],
                'rois': [dummy_roi],
                'cam_id': '',
                'tube_id': ''
        }

    def generate_actions(self):
        """
        Read tubes (imgs and rois) from self.cache, and feed to the NN for actions
        """
        if not self.cache:
            return []

        cache_underfilled = len(self.cache) < self.batch_size // 2
        if cache_underfilled and self.non_empty_cache_round < MAX_TUBE_AGE_IN_CACHE:
            self.non_empty_cache_round += 1
            return []

        self.log('cache size %d for NN' % len(self.cache))
        # Prepare the data for NN 
        self.non_empty_cache_round = 0
        imgs = []
        rois = []
        cam_ids = []
        tube_ids = []

        for i in range(self.batch_size):
            tube_data = self.cache.popleft() if self.cache else self.dummy_tube
            imgs.append(tube_data['imgs'])
            rois.append(tube_data['rois'][0])
            cam_ids.append(tube_data['cam_id'])
            tube_ids.append(tube_data['tube_id'])

        cur_time = time()
        probs = self.detect_on_tubes(imgs, rois)
        # self.log("proc time: %f" % (time() - cur_time))

        res = []
        for i in range(self.batch_size):
            # Skip dummy tube results 
            if not tube_ids[i]:
                break 
            top_classes = np.argsort(probs[i,:])[:-TOP_K_ACT-1:-1]
            log_act_prob_per_tube = []
            for t in range(TOP_K_ACT):
                class_id = top_classes[t]
                class_str = act.ACTION_STRINGS[class_id]
                class_prob = probs[i, class_id]
                # show the TOP ranked actions 
                log_act_prob_per_tube.append("%s:%.2f" % (class_str, class_prob))

                # Add the action if it's in the list and prob is higher than thres 
                if class_str in NN_ACT_THRES and NN_ACT_THRES[class_str] < class_prob:
                    res.append(Act(act_name=class_str, class1='person', tube1=tube_ids[i]))

            # self.log('ID(%s)-%s' % (tube_ids[i], log_act_prob_per_tube))

        return res 

    def detect_on_tubes(self, imgs, rois):
        """ 
        Detect the actions over the input imgs and rois as tubes 
        (1, 32, 400, 400, 3)
        @param: imgs: must be of shape batch_size x tube_len x 400 x 400 x 3
        @param: rois: must be of shape batch_size x tube_len x box
        @return: a list of prob of actions for each tube 
        """
        imgs = np.array(imgs)
        rois_np = np.array(rois)
        roi_batch_indices_np = np.arange(self.batch_size)

        # inputs
        input_seq_tf = self.detector_dict['input_seq']
        rois_tf = self.detector_dict['rois']
        roi_batch_indices_tf = self.detector_dict['roi_batch_indices']
        # output
        predictions_tf = self.detector_dict['pred_probs']

        feed_dict = {   input_seq_tf: imgs,
                        rois_tf: rois_np,
                        roi_batch_indices_tf: roi_batch_indices_np}

        act_detector = self.detector_dict['detector']
        return act_detector.session.run(predictions_tf, feed_dict=feed_dict)

    def set_up_detector(self, model_path):
        """ 
        init the NN and the input output dict 

        @param: model_path: the path to the model 
        """
        assert not self.detector_dict, "detector_dict alreay there!"

        act_detector = act.Action_Detector('soft_attn')
        input_seq, rois, roi_batch_indices, pred_probs = act_detector.define_inference_with_placeholders()
        act_detector.restore_model(model_path)

        self.detector_dict = {   
                                'detector':act_detector,
                                'input_seq': input_seq,
                                'rois': rois,
                                'roi_batch_indices': roi_batch_indices,
                                'pred_probs': pred_probs
                            }

    def run(self):
        ''' 
        Input: from in_queue, read ServerPkt (server.server_packet_manager)
        Output: to out_queue, write ServerPkt (server.server_packet_manager)
        '''
        self.set_up_detector(self.model_path)
        local_tube_cache = {}
        # self.vis = Visualizer()

        while True:
            server_pkt = self.in_queue.read()
            if server_pkt is None:
                sleep(0.01)
                continue

            for tube in server_pkt.tubes:
                if tube.label != 'person':
                    continue 

                tid = tube.tube_id
                if tid not in local_tube_cache:
                    local_tube_cache[tid] = deque()

                for clip in tube.tube_clips:
                    local_tube_cache[tid].append(clip)

                if len(local_tube_cache[tid]) < self.tube_size:
                    continue 

                tmp_tube_imgs = []
                tmp_tube_rois = []
                for _ in range(self.tube_size):
                    clip = local_tube_cache[tid].popleft()
                    tmp_tube_imgs.append(clip.img)
                    tmp_tube_rois.append(clip.roi)

                self.cache.append({'imgs': tmp_tube_imgs, 
                                    'rois': tmp_tube_rois,
                                    'cam_id': server_pkt.cam_id,
                                    'tube_id': tid})
                # self.log('add tube %s to cache' % str(tid))

            server_pkt.actions += self.generate_actions()
            self.out_queue.write(server_pkt)

        self.log('finish')


    def log(self, s):
        logging.debug('[NNAct] %s' % s)
