import tensorflow as tf
import numpy as np
import logging 


class TFDetector(object):
    def __init__(self, graph_path, label_file, session=None):
        ''' Use TF's obj detection API for detection 

        Args:
        - graph_path: file path to the pb model 
        - session: existing session
        '''
        self.graph_path = graph_path
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.detection_graph = detection_graph

        if not session:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.Session(graph=detection_graph, config=config)

        self.session = session

        self.label_mapping = {}
        lines = []
        with open(label_file, 'r') as fin:
            lines = fin.readlines()            

        for line in lines:
            line = line.strip()
            if not line or '#' in line:
                continue 
            id, label = line.split(' ')
            self.label_mapping[int(id)] = label

        self.log('init')


    def mapping_classes(self, classes):
        ''' Return the label name of all dets 
        '''
        return [[self.label_mapping[i] if i in self.label_mapping 
                            else str(i) for i in c] for c in classes]


    def detect_images(self, images):
        '''
        Runs the object detection on a single image or a batch of images.
        images can be a batch or a single image with batch dimension 1, 
        dims:[None, None, None, 3]

        Args:
        - images: a list of np array images 

        Return:
        - boxes: list of top, left, bottom, right
        - scores: list of confidence 
        - classes: list of labels 
        '''
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = self.session.run(
                                        [boxes, scores, classes, num_detections],
                                        feed_dict={image_tensor: images[:,:,:,:]})

        return boxes, scores, self.mapping_classes(classes)


    def log(self, s):
        logging.debug('[TFDetector] %s' % s)

