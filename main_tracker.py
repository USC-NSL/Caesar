from __future__ import absolute_import, division, print_function

'''
Function: read a video, run detection and send to next hop

Input: GRPC packet stream
        meta; {'box':[x0,y0,x1,y1], # in pixel position
            'score': confidence score of det in float,
            'label': name of the object in str}

Outputs: serialized GRPC packet (defined in network/data_packet)
            packet meta: a list of {'box':[x0,y0,x1,y1], # in pixel position
                                    'id': object track id,
                                    'label': name of the object in str}
'''

''' Preprocessing
'''
import logging 
import numpy as np
import os
import sys
from time import time, sleep
from multiprocessing import Process
from threading import Thread

import config.const_tracker as const
from tracker.deepsort import DeepSort
from tracker.feature_extractor import FExtractor
from tracker.reid import REID
from network.data_writer import DataWriter
from network.utils import Q
from network.socket_client import NetClient
from network.socket_server import NetServer


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
    server = NetServer(
        name='tracker',
        address=const.LOCAL_ADDR,
        port=const.LOCAL_PORT,
        buffer_size=const.QUEUE_SIZE
    )
    server_proc = Process(target=server.run)
    server_proc.start()

    client = NetClient(
        client_name='tracker',
        server_addr=const.SERVER_ADDR,
        buffer_size=const.QUEUE_SIZE
    )
    if const.UPLOAD_DATA:
        client_proc = Process(target=client.run)
        client_proc.start()

    feature_queue = Q(const.QUEUE_SIZE)
    feature_extractor = FExtractor(
        in_queue=server.data_queue,
        out_queue=feature_queue,
        model_path=const.TRACK_MODEL_PATH
    )
    tm_proc = Process(target=feature_extractor.run)
    tm_proc.start()

    reid = REID(topo_path=const.TOPO_PATH, img_shape=const.IMG_SHAPE)

    trackers = {}
    data_savers = {}
    print('Tracker init done!')

    cur_time = time()
    while running[0]:
        pkt = feature_queue.read()
        if pkt is None:
            sleep(0.01)
            continue

        cid = pkt.cam_id
        if cid not in trackers:
            trackers[cid] = DeepSort(
                track_labels=const.TRACK_LABELS, 
                attach_labels=const.ATTACH_LABELS
            )
            data_savers[cid] = DataWriter(
                file_path=RES_FOLDER+'{}.npy'.format(cid)
            )
            print('create tracker for video {}'.format(cid))

        pkt = trackers[cid].update(pkt)
        pkt = reid.update(pkt)

        if const.UPLOAD_DATA:
            client.send_data(pkt)

        if const.SAVE_DATA:
            data_savers[cid].save_data(frame_id=pkt.frame_id, meta=pkt.meta)

    if const.SAVE_DATA:
        for cid in data_savers:
            data_savers[cid].save_to_file()

    server.stop()
    print('tracker finished')


if __name__ == '__main__':
    logging.basicConfig(filename='tracker_debug.log',
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
