from __future__ import absolute_import, division, print_function

'''
Function: receive RPC pkts, show the result on webpage 

Input: GRPC packet stream, packet meta:
        a list of {'box':[x0,y0,x1,y1], # in pixel position
                    'id': object track id,
                    'label': name of the object in str,
                    'id2': (optional) the second tube id for an act,
                    'act_fid': the start frame id of an action}
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
from threading import Thread
from multiprocessing import Process
from web.code_generator import Generator
from web.visualizer import Visualizer
from network.data_writer import DataWriter
from network.socket_server import NetServer
from network.utils import Q
from network.video_manager import VideoWriter


SAVE_VIDEO = True 

LOCAL_NAME = 'web'
QUEUE_SIZE = 64
VIDEO_WRITE_FPS = 20

### CONFIG: True if you want to view the track id in the visualization
SHOW_TRACK = True 

### CONFIG: Make these same as your video's setting
VIDEO_WRITE_WID = 640
VIDEO_WRITE_HEI = 480

### CONFIG: address and port for incoming traffic
LOCAL_ADDR = 'localhost'
LOCAL_PORT = 50053

### CONFIG: must contain all mobile names (e.g. if you have two
# devices v1 and v2, you should put both of them here)
CLIENT_NAMES = ['v1', 'v2']

### CONFIG: url for webpage, change 'localhost' to the server's IP
WEB_ADDR = 'localhost:50088' 
### CONFIG: the display FPS on the webpage. This should be same as mobile's sFPS
WEB_DISPLAY_FPS = 15


def main(running):
    server = NetServer(
                        name=LOCAL_NAME, 
                        address=LOCAL_ADDR,
                        port=LOCAL_PORT,
                        buffer_size=QUEUE_SIZE
                    ) 
    server_proc = Process(target=server.run)
    server_proc.start()
    
    vis = Visualizer()

    code_generator = Generator(CLIENT_NAMES, (VIDEO_WRITE_WID, VIDEO_WRITE_HEI))
    code_generator.run()

    web_msg_queue = Q(2)           # data from the webpage (user input)
    display_queues = {v: Q(QUEUE_SIZE) for v in CLIENT_NAMES}  # image queues

    from web.web_server import WebServer
    ip, port = WEB_ADDR.split(':')
    wserver = WebServer(data_queues=display_queues, 
                        msg_queue=web_msg_queue, 
                        ip=ip, port=int(port),
                        display_fps=WEB_DISPLAY_FPS)

    wserver_thread = Thread(target=wserver.run)     
    wserver_thread.setDaemon(True)
    wserver_thread.start()        

    video_writers = {}
    if SAVE_VIDEO:
        video_writers = {v : VideoWriter(fname=RES_FOLDER + '%s.avi' % v,
                                        fps=VIDEO_WRITE_FPS, 
                                        resolution=(VIDEO_WRITE_WID, 
                                                    VIDEO_WRITE_HEI))
                                        for v in CLIENT_NAMES
                        }

    while running:
        d = server.read_data()
        if d is None:
            sleep(0.01)
            continue

        img = d.img
        vname = d.cam_id
        if SAVE_VIDEO:
            assert vname in video_writers, "this src not registered!"
            video_writers[vname].save_frame(img)

        for m in d.meta:
            if 'act_fid' in m:
                vis.reg_act(cam_id=vname, data=m)
            else:
                vis.draw_track(img=img, data=m, cam_id=vname, show=SHOW_TRACK)
        vis.draw_act(img=img, cam_id=vname)
        display_queues[vname].write(img) 
        
    if SAVE_VIDEO:
        for v in video_writers:
            video_writers[v].close()
    server.stop()

    print('web finished')


if __name__ == '__main__':
    logging.basicConfig(filename='web_debug.log',
                        format='%(asctime)s %(message)s',
                        datefmt='%I:%M:%S  ',
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
