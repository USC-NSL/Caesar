from __future__ import absolute_import, division, print_function

'''
Function: receive pkts, run act detection and send to next hop

Input: GRPC packet stream, packet meta:
        a list of {'box':[x0,y0,x1,y1], # in pixel position
                    'id': object track id,
                    'label': name of the object in str}

Internal data: ServerPkt (in server/server_packet.py)

Outputs: serialized GRPC packet (defined in network/data_packet)
        packet meta: a list of {'box':[x0,y0,x1,y1], # in pixel position
                                'id': object track id, (re-ided)
                                'label': name of the object in str}
'''


'''
Preprocessing
'''
import sys
import os

RES_FOLDER = 'res/{}/'.format(os.path.basename(__file__).split('.py')[0])
if not os.path.exists(RES_FOLDER):
    os.makedirs(RES_FOLDER)
print('Output to {}'.format(RES_FOLDER))


'''
Import packages
'''
import logging

from time import time, sleep
from multiprocessing import Process
from threading import Thread

from server.server_packet_manager import ServerPktManager
from server.action_spatial import SpatialActDetector
from server.action_nn import NNActDetector
from server.action_comp import CompActDetector
from network.data_writer import DataWriter
from network.socket_client import NetClient
from network.socket_server import NetServer
from network.utils import Q


''' Configuration area, change these values on your demand
'''
SAVE_DATA = True
UPLOAD_DATA = True

LOCAL_NAME = 'acter'
QUEUE_SIZE = 32
TRACK_LABELS = ['person', 'car']
ATTACH_LABELS = ['bike', 'bag']

NN_BATCH = 4
TUBE_SIZE = 32

### CONFIG: address and port for incoming traffic
LOCAL_ADDR = 'localhost'
LOCAL_PORT = 50052

### CONFIG: server address for next hop 
SERVER_ADDR = 'localhost:50053'

### CONFIG: path to the action detection NN model 
NN_ACT_MODEL_PATH = 'checkpoints/model_ckpt_soft_attn_pooled_cosine_drop_ava-130'


'''
Main function
'''
def main(running):
    server = NetServer(name=LOCAL_NAME,
                        address=LOCAL_ADDR,
                        port=LOCAL_PORT,
                        buffer_size=QUEUE_SIZE)
    server_proc = Process(target=server.run)
    server_proc.start()

    client = NetClient(client_name=LOCAL_NAME,
                        server_addr=SERVER_ADDR,
                        buffer_size=QUEUE_SIZE)
    if UPLOAD_DATA:
        client_proc = Process(target=client.run)
        client_proc.start()

    data_savers = {}

    tube_queue = Q(QUEUE_SIZE)
    act_spatial_queue = Q(QUEUE_SIZE)
    act_nn_queue = Q(QUEUE_SIZE)
    act_comp_queue = Q(QUEUE_SIZE)
    filter_queue = Q(QUEUE_SIZE)

    tm = ServerPktManager(
                            in_queue=server.data_queue,
                            out_queue=tube_queue,
                            track_list=TRACK_LABELS,
                            overlap_list=ATTACH_LABELS
                        )
    tm_proc = Thread(target=tm.run)
    tm_proc.deamon = True
    tm_proc.start()

    spatial_act = SpatialActDetector(
                                in_queue = tube_queue,
                                out_queue=act_spatial_queue,
                            )
    spatial_proc = Process(target=spatial_act.run)
    spatial_proc.deamon = True
    spatial_proc.start()

    nn_act = NNActDetector(in_queue=act_spatial_queue,
                            out_queue=act_nn_queue,
                            model_path=NN_ACT_MODEL_PATH,
                            batch_size=NN_BATCH,
                            tube_size=TUBE_SIZE,
                            filter_queue=filter_queue,
                            )
    nn_act_proc = Process(target=nn_act.run)
    nn_act_proc.deamon = True
    nn_act_proc.start()


    comp_act = CompActDetector(in_queue=act_nn_queue,
                                out_queue=act_comp_queue,
                                filter_queue=filter_queue,
                            )
    comp_act_proc = Process(target=comp_act.run)
    comp_act_proc.deamon = True
    comp_act_proc.start()


    out_queue = act_comp_queue
    print('server starts')

    while running[0]:
        server_pkt = out_queue.read()
        if server_pkt is None:
            sleep(0.01)
            continue

        cid = server_pkt.cam_id
        if cid not in data_savers:
            data_savers[cid] = DataWriter(file_path=RES_FOLDER+'{}.npy'.format(cid))
            print('creat data_saver for video {}'.format(cid))

        logging.info("Cam-{}, Frame-{}, Acts-{}".format(
                                                server_pkt.cam_id,
                                                server_pkt.get_first_frame_id(),
                                                server_pkt.get_action_info()
                                                ))
        logging.info("--------------------------------")

        data_pkts = server_pkt.to_data_pkts()
        for p in data_pkts:
            if UPLOAD_DATA:
                client.send_data(p)

            if SAVE_DATA:
                data_savers[cid].save_data(frame_id=p.frame_id, meta=p.meta)

    if SAVE_DATA:
        for cid in data_savers:
            data_savers[cid].save_to_file()

    server.stop()
    print("server finished!")


if __name__ == '__main__':
    logging.basicConfig(filename='act_debug.log',
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
