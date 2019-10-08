from __future__ import absolute_import, division, print_function

'''
Function: read a video, run detection and send to next hop

Params: video path

Outputs: serialized GRPC packet (defined in network/data_packet)
            packet meta: a list of {'box':[x0,y0,x1,y1], # in pixel position
                                    'score': confidence score of det in float,
                                    'label': name of the object in str}
'''

import sys
import os
import numpy as np
import logging 
from time import time, sleep
from multiprocessing import Process
from threading import Thread

import config.const_mobile as const
from network.data_reader import DataReader
from network.data_writer import DataWriter
from network.socket_client import NetClient
from network.data_packet import DataPkt


"""
Create result folder
"""
RES_FOLDER = 'res/{}/'.format(os.path.basename(__file__).split('.py')[0])
if not os.path.exists(RES_FOLDER):
    os.makedirs(RES_FOLDER)
print('Output to {}'.format(RES_FOLDER))


"""
Main function
"""
def main(running):
    reader = DataReader(video_path=const.VIDEO_PATH,
                        file_path='',
                        max_frame_id=const.MAX_FRAME_ID)

    detector = None 
    if const.OBJ_MODEL == 'mobilenet':
        from mobile.object_detector_tf import TFDetector
        detector = TFDetector(
            graph_path=const.OBJ_MODEL_PATH, 
            label_file=const.OBJ_LABEL_FILE,
        )
    elif const.OBJ_MODEL == 'mrcnn':
        '''
        mrcnn_path = os.path.join(os.getcwd(), 'mobile/maskrcnn_benchmark')
        print("mask rcnn path", mrcnn_path)
        sys.path.append(mrcnn_path)
        '''
        from mobile.object_detector_mrcnn_torch import MRCNN
        detector = MRCNN(model_path=const.OBJ_MODEL_PATH)
    else:
        raise ValueError("Model not implemented!")

    uploader = NetClient(
        client_name=const.CLIENT_NAME,
        server_addr=const.SERVER_ADDR,
        buffer_size=const.QUEUE_SIZE,
    )

    if const.UPLOAD_DATA:
        uploader_proc = Process(target=uploader.run)
        uploader_proc.start()

    data_saver = DataWriter(
        file_path=RES_FOLDER + '{}.npy'.format(const.CLIENT_NAME)
    )

    print('Mobile init done!')

    frame_cache = []
    frame_id_cache = []

    timer1 = time()
    timer2 = time()
    time_gap = float(const.OBJ_BATCH_SIZE) / float(const.UPLOAD_FPS)

    while running[0]:
        img, frame_id, meta = reader.get_data()
        if not len(img):
            break

        # Print frame id and FPS every 20 frames 
        if not frame_id % const.OBJ_BATCH_SIZE:
            print('frame {}, avg FPS {}'.format(
                    frame_id,
                    const.OBJ_BATCH_SIZE / (time()-timer1),
                )
            )
            timer1 = time()

        frame_cache.append(img)
        frame_id_cache.append(frame_id)
        if len(frame_cache) < const.OBJ_BATCH_SIZE:
            continue

        boxes, scores, classes = detector.detect_images(
            np.stack(frame_cache, axis=0)
        )

        H, W, _ = img.shape
        for i in range(const.OBJ_BATCH_SIZE):
            meta = []
            for j in range(len(boxes[i])):
                if scores[i][j] < const.OBJ_THRES:
                    continue
                b = boxes[i][j]
                meta.append({'box': [int(b[1]*W), int(b[0]*H),
                                    int(b[3]*W), int(b[2]*H)],
                            'label': classes[i][j],
                            'score': scores[i][j]})

            pkt = DataPkt(
                img=frame_cache[i], 
                cam_id=const.CLIENT_NAME,
                frame_id=frame_id_cache[i], 
                meta=meta,
            )

            if const.UPLOAD_DATA:
                uploader.send_data(pkt)

            if const.SAVE_DATA:
                data_saver.save_data(
                    frame_id=pkt.frame_id, 
                    meta=pkt.meta,
                )

        time_past = time() - timer2
        sleep_time = max(0, time_gap - time_past)
        sleep(sleep_time)
        timer2 = time()

        frame_cache = []
        frame_id_cache = []

    if const.SAVE_DATA:
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
