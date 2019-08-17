import pickle
import cv2
import numpy as np


class DataPkt:
    def __init__(self, img=None, cam_id='', frame_id=0, meta=[]):
        ''' Pkt to be tranismitted between machines

        Args:
        - img: the cv2 img
        - cam_id: str
        - frame_id: int

        - meta: a list of bounding boxes with following fields:
            "box": [x0, y0, x1, y1]
            "id": track_id (str) or the id1 of an act
            "label": str, class of the object/act
            "act_fid": start frame id of the action [enabled only for acts!]
            "id2": (str) another id as act subject [enabled for acts, optional]
            "feature": list, reid feature
            "score": float, confidence of the box
        '''
        self.img = img
        self.cam_id = cam_id
        self.frame_id = frame_id
        self.meta = meta

    def encode_img(self, img):
        # return b64encode(cv2.imencode('.jpg', im)[1].tostring())
        return cv2.imencode('.jpg', img)[1].tostring()

    def decode_img(self, data):
        d = np.fromstring(data, dtype=np.uint8)
        return cv2.imdecode(d, 1)

    def load_from_string(self, s):
        d = pickle.loads(s)
        self.img = self.decode_img(d['img'])
        self.cam_id = d['cam_id']
        self.frame_id = d['frame_id']
        self.meta = d['meta']

    def encode(self):
        ''' return a serialized data for data streaming
        '''
        output = {  'img': self.encode_img(self.img),
                    'frame_id': self.frame_id,
                    'cam_id': self.cam_id,
                    'meta': self.meta,
                }

        return pickle.dumps(output)
