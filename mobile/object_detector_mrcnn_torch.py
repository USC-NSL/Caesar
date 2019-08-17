import logging 
import os 

from maskrcnn_benchmark.config import cfg
from mobile.mrcnn_pytorch.predictor import COCODemo


CONFIG_FILE = "e2e_faster_rcnn_X_101_32x8d_FPN_1x_visdrone.yaml"
CONFIG_PATH = os.path.join(os.getcwd(), "mobile", "mrcnn_pytorch", CONFIG_FILE)

CONFIDENCE_THRESHOLD = 0.7
IMG_SIZE = 800


class MRCNN:
    def __init__(self, model_path):
        model_file = os.path.join(os.getcwd(), model_path)
        assert os.path.exists(model_file), 'model_file not exists!'
        assert os.path.exists(CONFIG_PATH), 'config file not exists!'

        cfg.merge_from_file(CONFIG_PATH)
        cfg.merge_from_list(["MODEL.WEIGHT", model_file])

        self.coco_demo = COCODemo(
            cfg,
            min_image_size=IMG_SIZE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )

        self.label_mapping = {
            'pedestrian' : 'person', 
            'van' : 'car',
            'truck' : 'car',
            'bus' : 'car',
            'tricycle' : 'motorcycle', 
            'motor' : 'motorcycle', 
            'awning-tricycle' : 'motorcycle', 
        }

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
            H, W, _ = img.shape

            res = self.coco_demo.compute_prediction(img)
            predictions = self.coco_demo.select_top_predictions(res)

            tmp_boxes = predictions.bbox.tolist()
            tmp_boxes = [[b[1] / H, b[0] / W, b[3] / H, b[2] / W] for b in tmp_boxes]
            tmp_scores = predictions.get_field("scores").tolist()
            tmp_classes = []
            for label in predictions.get_field("labels").tolist():
                l = self.coco_demo.CATEGORIES[label]
                if l in self.label_mapping: 
                    tmp_classes.append(self.label_mapping[l])
                else:
                    tmp_classes.append(l) 

            boxes.append(tmp_boxes)
            scores.append(tmp_scores)
            classes.append(tmp_classes)

        return boxes, scores, classes


    def log(self, s):
        logging.debug('[MRCNN] %s' % s)
