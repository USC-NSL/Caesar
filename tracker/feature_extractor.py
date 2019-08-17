import numpy as np
from time import time, sleep
import logging 
from tracker.deep_sort.generate_detections import create_box_encoder


def feature_distance(f1, f2):
    # f1 and f2 should be normalized
    a = np.asarray(f1)
    b = np.asarray(f2)
    return 1. - np.dot(a, b.T)

class FExtractor:
    def __init__(self, in_queue, out_queue, model_path):
        ''' Generate feature for person dets in pkt. Read data from in_queue,
            and write the pkt with feature to out_queue
        '''
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.model_path = model_path
        self.log('init')


    def run(self):
        ''' Keeps the extractor running and generate features
            Read pkt from in_queue and write modified pkt to out_queue
        '''
        encoder = create_box_encoder(self.model_path, batch_size=16)

        while True:
            pkt = self.in_queue.read()
            if pkt is None:
                sleep(0.01)
                continue

            ds_boxes = []
            person_indices = []
            for i, info in enumerate(pkt.meta):
                if info['label'] != 'person':
                    continue
                b = info['box']
                person_indices.append(i)
                ds_boxes.append([b[0], b[1], b[2] - b[0], b[3] - b[1]])

            features = encoder(pkt.img, ds_boxes)
            for i in range(len(features)):
                pkt.meta[person_indices[i]]['feature'] = features[i]

            self.out_queue.write(pkt)


    def log(self, s):
        logging.debug('[FExtractor] %s' % s)
