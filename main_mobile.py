from __future__ import absolute_import, division, print_function

'''
Function: read a video, run detection and send to next hop

Params: video path

Outputs: serialized GRPC packet (defined in network/data_packet)
            packet meta: a list of {'box':[x0,y0,x1,y1], # in pixel position
                                    'score': confidence score of det in float,
                                    'label': name of the object in str}
'''

''' Preprocessing
'''
import sys
import os

RES_FOLDER = 'res/{}/'.format(os.path.basename(__file__).split('.py')[0])
if not os.path.exists(RES_FOLDER):
    os.makedirs(RES_FOLDER)
print('Output to {}'.format(RES_FOLDER))


''' Import packages
'''
import numpy as np
from time import time, sleep
from multiprocessing import Process
from threading import Thread
import logging 

from network.data_reader import DataReader
from network.data_writer import DataWriter
from network.socket_client import NetClient
from network.data_packet import DataPkt


''' Configuration area, change these values on your demand
'''
SAVE_DATA = True
UPLOAD_DATA = True
UPLOAD_FPS = 20

MAX_FRAME_ID = -1
QUEUE_SIZE = 64

OBJ_LABEL_FILE = 'config/label_mapping.txt'
OBJ_THRES = 0.25

### CONFIG: path to the video source 
VIDEO_PATH = 'data/v1.avi' # '0'
### CONFIG: name the mobile client 
CLIENT_NAME = 'v1'
### CONFIG: server address 
SERVER_ADDR = 'localhost:50051'
### CONFIG: choose which obj detector to use 
OBJ_MODEL = 'mobilenet' 
# OBJ_MODEL = 'yolo'
# OBJ_MODEL = 'mrcnn'
### CONFIG: path to obj detection model (should match the OBJ_MODEL)
OBJ_MODEL_PATH = 'checkpoints/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'   # mobilenet
# OBJ_MODEL_PATH = 'checkpoints/visdrone_model_0360000.pth'                                 # mrcnn
### CONFIG: specify the batch size for obj detector (recommend: 16 for mobilenet, 1 for others)
OBJ_BATCH_SIZE = 16


''' Main function
'''
def main(running):
    reader = DataReader(video_path=VIDEO_PATH,
                        file_path='',
                        max_frame_id=MAX_FRAME_ID)

    detector = None 
    if OBJ_MODEL == 'yolo':
        from mobile.object_detector_yolo import YOLO
        detector = YOLO()
    elif OBJ_MODEL == 'mobilenet':
        from mobile.object_detector_tf import TFDetector
        detector = TFDetector(graph_path=OBJ_MODEL_PATH, label_file=OBJ_LABEL_FILE)
    elif OBJ_MODEL == 'mrcnn':
        from mobile.object_detector_mrcnn_torch import MRCNN
        detector = MRCNN(model_path=OBJ_MODEL_PATH)
    else:
        raise ValueError("Model not implemented!")

    uploader = NetClient(client_name=CLIENT_NAME,
                        server_addr=SERVER_ADDR,
                        buffer_size=QUEUE_SIZE)
    if UPLOAD_DATA:
        uploader_proc = Process(target=uploader.run)
        uploader_proc.start()

    data_saver = DataWriter(file_path=RES_FOLDER+'{}.npy'.format(CLIENT_NAME))

    time_gap = float(OBJ_BATCH_SIZE) / float(UPLOAD_FPS)
    print('Mobile init done!')

    frame_cache = []
    frame_id_cache = []

    timer1 = time()
    timer2 = time()
    while running[0]:
        img, frame_id, meta = reader.get_data()
        if not len(img):
            break

        if not frame_id % 20:
            print('frame {}, avg FPS {}'.format(frame_id,
                                                20 / (time()-timer1)))
            timer1 = time()

        frame_cache.append(img)
        frame_id_cache.append(frame_id)
        if len(frame_cache) < OBJ_BATCH_SIZE:
            continue

        boxes, scores, classes = detector.detect_images(np.stack(frame_cache, axis=0))

        H, W, _ = img.shape
        for i in range(OBJ_BATCH_SIZE):
            meta = []
            for j in range(len(boxes[i])):
                if scores[i][j] < OBJ_THRES:
                    continue
                b = boxes[i][j]
                meta.append({'box': [int(b[1]*W), int(b[0]*H),
                                    int(b[3]*W), int(b[2]*H)],
                            'label': classes[i][j],
                            'score': scores[i][j]})

            pkt = DataPkt(img=frame_cache[i], cam_id=CLIENT_NAME,
                            frame_id=frame_id_cache[i], meta=meta)

            if UPLOAD_DATA:
                uploader.send_data(pkt)

            if SAVE_DATA:
                data_saver.save_data(frame_id=pkt.frame_id, meta=pkt.meta)

        time_past = time() - timer2
        sleep(max(0, time_gap - time_past))
        timer2 = time()

        frame_cache = []
        frame_id_cache = []

    if SAVE_DATA:
        data_saver.save_to_file()


if __name__ == '__main__':
    logging.basicConfig(filename='mobile_debug.log',
                        format='%(asctime)s %(message)s',
                        datefmt='%I:%M:%S ',
                        filemode='w',
                        level=logging.DEBUG)
    running = [True]
    th = Thread(target=main, args=(running,))
    th.start()
    while True:
        try:
            sleep(10)
        except (KeyboardInterrupt, SystemExit):
            running[0] = False
            break
    print('done')
