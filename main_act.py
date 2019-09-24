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


import sys
import os
import logging
from time import time, sleep
from multiprocessing import Process
from threading import Thread

import config.const_act as const
from server.server_packet_manager import ServerPktManager
from server.action_spatial import SpatialActDetector
from server.action_nn import NNActDetector
from server.action_comp import CompActDetector
from network.data_writer import DataWriter
from network.socket_client import NetClient
from network.socket_server import NetServer
from network.utils import Q


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
        name='act',
        address=const.LOCAL_ADDR,
        port=const.LOCAL_PORT,
        buffer_size=const.QUEUE_SIZE,
    )
    server_proc = Process(target=server.run)
    server_proc.start()

    client = NetClient(
        client_name='act',
        server_addr=const.SERVER_ADDR,
        buffer_size=const.QUEUE_SIZE,
    )
    if const.UPLOAD_DATA:
        client_proc = Process(target=client.run)
        client_proc.start()

    data_savers = {}

    tube_queue = Q(const.QUEUE_SIZE)
    act_spatial_queue = Q(const.QUEUE_SIZE)
    act_nn_queue = Q(const.QUEUE_SIZE)
    act_comp_queue = Q(const.QUEUE_SIZE)
    filter_queue = Q(const.QUEUE_SIZE)

    tm = ServerPktManager(
        in_queue=server.data_queue,
        out_queue=tube_queue,
        track_list=const.TRACK_LABELS,
        overlap_list=const.ATTACH_LABELS,
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

    nn_act = NNActDetector(
        in_queue=act_spatial_queue,
        out_queue=act_nn_queue,
        model_path=const.NN_ACT_MODEL_PATH,
        batch_size=const.NN_BATCH,
        tube_size=const.TUBE_SIZE,
        filter_queue=filter_queue,
    )
    nn_act_proc = Process(target=nn_act.run)
    nn_act_proc.deamon = True
    nn_act_proc.start()


    comp_act = CompActDetector(
        in_queue=act_nn_queue,
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
            if const.UPLOAD_DATA:
                client.send_data(p)

            if const.SAVE_DATA:
                data_savers[cid].save_data(frame_id=p.frame_id, meta=p.meta)

    if const.SAVE_DATA:
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
