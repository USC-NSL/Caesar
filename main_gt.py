from __future__ import absolute_import, division, print_function

'''
Function: read several videos and their metadata, visualize, and send to next hop

Params: video path, and stored detection results

Outputs: serialized GRPC packet (defined in network/data_packet)
            packet meta: a list of {'box':[x0,y0,x1,y1], # in pixel position
                                    'score': confidence score of det in float,
                                    'label': name of the object in str,
                                    'id': track id (optional)}
'''


import logging
import sys
import os
import cv2
from time import time, sleep
from multiprocessing import Process
from threading import Thread

import config.const_gt as const
from network.data_reader import DataReader
from network.video_manager import VideoWriter
from network.utils import fname
from network.socket_client import NetClient
from network.data_packet import DataPkt
from web.visualizer import Visualizer


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
    if len(const.VIDEO_LIST) != len(const.META_LIST):
        print('error: inconsistent input data list!')
        return

    print('read meta from: %s' % str(const.META_LIST))
    readers = {}
    timers = {}
    video_writers = {}
    video_names = []

    for i in range(len(const.META_LIST)):
        n = fname(const.VIDEO_LIST[i])
        readers[n] = DataReader(
            video_path=const.VIDEO_LIST[i], 
            file_path=const.META_LIST[i],
        )

        timers[n] = time()
        video_names.append(n)
        if const.SAVE_VIDEO:
            video_writers[n] = VideoWriter(
                fname=RES_FOLDER+'{}.avi'.format(n),
                fps=const.VIDEO_WRITE_FPS,
                resolution=const.IMG_SHAPE,
            )

    uploader = NetClient(
        client_name='gt', 
        server_addr=const.SERVER_ADDR,
        buffer_size=const.QUEUE_SIZE,
    )
    if const.UPLOAD_DATA:
        uploader_proc = Process(target=uploader.run)
        uploader_proc.start()

    vis = Visualizer()

    time_gap = 1. / float(const.VIDEO_READ_FPS)
    print('Mobile init done!')

    ended_video_numbers = set()   # a list of video names that finished reading
    video_ind = 0
    while running[0] and len(ended_video_numbers) < len(video_names):
        if video_ind == len(video_names):
            video_ind = 0
            continue

        if video_ind in ended_video_numbers:
            video_ind += 1
            continue

        vname = video_names[video_ind]
        img, frame_id, meta = readers[vname].get_data()

        if not len(img):
            print('video %s has ended' % vname)
            ended_video_numbers.add(video_ind)
            video_ind += 1
            continue

        if not frame_id % 30:
            print('%s: frame %d' % (vname, frame_id))

        if const.UPLOAD_DATA:
            pkt = DataPkt(img=img, cam_id=vname, frame_id=frame_id, meta=meta)
            uploader.send_data(pkt)

        img2 = None
        if const.SAVE_VIDEO or const.SHOW_VIDEO:
            img2 = img.copy()
            vis.draw_frame_id(img2, vname, frame_id)
            for m in meta:
                if 'act_fid' in m:
                    vis.reg_act(cam_id=vname, data=m)
                else:
                    vis.draw_track(img=img2, data=m, cam_id=vname, show=const.SHOW_TRACK)
            vis.draw_act(img=img2, cam_id=vname)

        if const.SAVE_VIDEO:
            video_writers[vname].save_frame(img2)

        if const.SHOW_VIDEO:
            cv2.imshow(vname, img2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break

        time_past = time() - timers[vname]
        sleep(max(0, time_gap - time_past))
        timers[vname] = time()

        video_ind += 1

    if const.SAVE_VIDEO:
        for v in video_writers:
            video_writers[v].close()

    print('gt finished')


if __name__ == '__main__':
    logging.basicConfig(filename='gt_debug.log',
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
