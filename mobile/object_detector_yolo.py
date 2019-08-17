# Darkflow should be installed from: https://github.com/thtrieu/darkflow
import numpy as np
from time import time, sleep
from darkflow.net.build import TFNet
import logging 
from os.path import join 
import os 


# Place your downloaded cfg and weights under "checkpoints/"
YOLO_CONFIG = join(os.getcwd(),'checkpoints/yolo_cfg')
YOLO_MODEL = join(os.getcwd(),'checkpoints/yolo_cfg/yolo.cfg')
YOLO_WEIGHTS = join(os.getcwd(),'checkpoints/yolo.weights')

GPU_ID = 0
GPU_UTIL = 0.5
YOLO_THRES = 0.4
DETECT_PERSON_ONLY = True


class YOLO:
    def __init__(self):
        ''' Use YOLO for detection 

        Args:
        - graph_path: path to the folder that saves yolo config and model 
        - label_file: NA 
        '''

        opt = { "config": YOLO_CONFIG,
                "model": YOLO_MODEL,  
                "load": YOLO_WEIGHTS,
                "gpuName": GPU_ID,
                "gpu": GPU_UTIL,
                "threshold": YOLO_THRES
            }
        self.tfnet = TFNet(opt)
        self.label_mapping = {'handbag':'bag', 'backpack':'bag'}
        self.log('init')


    def detect_images(self, images):
        '''
        Runs the object detection on a batch of images.
        images can be a batch or a single image with batch dimension 1, 
        dims:[None, None, None, 3]

        Args:
        - images: a list of np array images 

        Return:
        - boxes: list of top, left, bottom, right (in ratio)
        - scores: list of confidence 
        - classes: list of labels 
        '''
        boxes = []
        scores = []
        classes = []

        for img in images:
            dets = self.tfnet.return_predict(img)
            tmp_boxes = []
            tmp_scores = []
            tmp_classes = []
            h, w, _ = img.shape

            for d in dets:
                if DETECT_PERSON_ONLY and d['label'] != 'person':
                    continue 

                tmp_boxes.append([d['topleft']['y']/h, d['topleft']['x']/w,
                                d['bottomright']['y']/h, d['bottomright']['x']/w])
                tmp_scores.append(d['confidence'])

                label = d['label'] 
                if d['label'] in self.label_mapping:
                    label = self.label_mapping[d['label']]
                tmp_classes.append(label)

            boxes.append(tmp_boxes)
            scores.append(tmp_scores)
            classes.append(tmp_classes)

        return boxes, scores, classes


    def log(self, s):
        logging.debug('[YOLO] %s' % s)
