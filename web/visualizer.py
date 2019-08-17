import cv2    
import logging  
from collections import defaultdict 


ACTION_DURATION = 20
class ActRecordManager:
    def __init__(self, max_act_age=ACTION_DURATION):
        self.record = defaultdict(dict)
        self.max_act_age = max_act_age

    def add_act(self, cam_id, id1, label, id2=''):
        self.record[cam_id][id1, id2, label] = 0

    def get_acts(self, cam_id):
        res = []
        deleted_keys = []
        for key, age in self.record[cam_id].items():
            if age < self.max_act_age:
                res.append(key)
                self.record[cam_id][key] += 1
            else:
                deleted_keys.append(key)

        for key in deleted_keys:
            del self.record[cam_id][key]

        return res 


class TubeBoxManager:
    def __init__(self):
        self.record = defaultdict(dict)

    def update_tube(self, cam_id, id, box):
        self.record[cam_id][id] = box

    def get_tube_box(self, cam_id, id):
        return self.record[cam_id][id] if id in self.record[cam_id] else []


class Visualizer:
    def __init__(self):
        self.colors = [ 
                        (255,0,0), (0,0,255), (0, 255,0),
                        (0,255,255), (255,0,255), (255,255,0),
                        (150,255,107), (75,187,107), (225,10,179), 
                        (187,107,215), (150,225,90), (200,120,225), 
                    ]
        self.line_margin_pix = 10 
        self.line_head_margin_y = 29
        self.line_head_margin_x = 20
        self.headline_color = (255,255,255)

        self.act_record_manager = ActRecordManager()
        self.tube_box_manager = TubeBoxManager()

    def draw_box(self, img, box, label, color_id=0):    # draw bbox and name
        '''
        Args:
        - box: [left, top, right, bottom]
        - img: image 
        - label: text label of the image 

        Return: NA
        '''
        color = self.colors[color_id % len(self.colors)] if color_id >= 0 else (255,255,255)
        # print(box)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        self.text(txt=str(label), pos=(box[0], box[1] - 10), img=img, color=color)

    def log(self, s):
        logging.debug('[Visualizer] %s' % s )

    def headlines(self, img, labels):
        """
        Show each detected actions at the top of frames 
        """
        start_x, start_y = self.line_head_margin_x, self.line_head_margin_y
        for l in labels:
            text_width, text_height = cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 
                                                    fontScale=1, thickness=3)[0]
            box_coords = ((start_x - 3, start_y + 8), (start_x + text_width - 7, 
                                                    start_y - text_height - 2))
            cv2.rectangle(img, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)

            self.text(txt=l, pos=(start_x, start_y), img=img, 
                            color=self.headline_color, fsize=0.7, thickness=2)
            start_y += self.line_head_margin_y

    def text(self, txt, pos, img, color=(255,255,255), fsize=0.6, thickness=2):
        cv2.putText(img, txt, (pos[0], pos[1]), cv2.FONT_HERSHEY_SIMPLEX, fsize, 
                                                                color, thickness)

    def show(self, img):
        cv2.imshow('res', img) 
        cv2.waitKey(1)

    def draw_frame_id(self, img, cam_id, frame_id):
        ''' Draw a frame with camera id and frame id

        Input: 
        - img: cv2 frame 
        - cam_id: camera id 
        - frame_id: frame number
        '''
        self.text(txt='%s: %d' % (cam_id, frame_id), pos=(30, 30), img=img, 
                                                            color=(125,255,200))

    def draw_act(self, img, cam_id):
        """
        Draw all actions for the cam_id based on each act's tube's last positions
        """
        res = []
        labels = []
        for id1, id2, lb in self.act_record_manager.get_acts(cam_id):
            box1 = self.tube_box_manager.get_tube_box(cam_id, id1)
            if not box1:
                continue 

            if id2:
                box2 = self.tube_box_manager.get_tube_box(cam_id, id2)
                if not box2:
                    continue 

                box1 = [min(box1[0], box2[0]), 
                        min(box1[1], box2[1]), 
                        max(box1[2], box2[2]), 
                        max(box1[3], box2[3]),]

            label = id1 + ':' + lb
            if id2:
                label += ':' + id2

            self.draw_box(img, box=box1, label=label, color_id=ord(lb[0])) 
            labels.append(label)
        self.headlines(img, labels)

    def reg_act(self, cam_id, data):
        """
        Register the action so we can display it in following frames
        
        Input: 
        - cam_id: camera id of the metadata 
        - data: {   
                    'id':tube_id, 
                    'label':label,
                    'act_fid':start_fid_of_the_act, 
                    'id2':tube_id2 (optional)
                }
        """
        assert 'act_fid' in data, "Should not use draw_act for tracks!"
        self.act_record_manager.add_act(
                                            cam_id=cam_id,
                                            id1=data['id'],
                                            label=data['label'],
                                            id2=data['id2'] if 'id2' in data else '',
                                        )

    def draw_track(self, img, data, cam_id='', show=True):
        ''' 
        Draw a frame with frame id, track boxes, and labels 

        Input: 
        - img: cv2 frame 
        - data: {'box': [x0,y0,x1,y1], 'id':tube_id, 'label':label}
        '''
        assert 'act_fid' not in data, "Should not use draw_boxes for action!"
        if 'id' in data:   # trackable objs 
            if show:
                self.draw_box(img, box=data['box'], label=data['id'], color_id=data['id'])
            act_tube_id = data['label'] + '-' + str(data['id'])
            self.tube_box_manager.update_tube(cam_id, act_tube_id, data['box'])
        else:           # un-trackable objs
            if show:
                self.draw_box(img, box=data['box'], label=data['label'], color_id=ord(data['label'][0]))

    def draw_traces(self, img, traces):
        ''' Draw a frame with track traces 

        Input: 
        - img: cv2 frame 
        - traces: [{'tube_id':tube_id, 'trace':list of [x0,y0,x1,y1]}]
        '''
        for trace in traces:
            for i in range(len(trace['trace']) - 1):
                self.draw_line(img, p1=trace['trace'][i], p2=trace['trace'][i+1],
                                                        color_id=trace['id'])