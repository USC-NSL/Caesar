from threading import Thread
from time import sleep
import sys 
import logging 
from network.utils import Q
from web.code_generator import Generator


class Web:
    def __init__(self, web_address, queue_size, video_names, video_wid, video_hei):
        '''
        Args:
        - web_address: ipv4:port
        - queue_size: size of img queue for each display
        - video_names: a full list of video names (client names)
        - video_wid, video_hei: size you want to display image

        Return: NA
        '''
        self.web_address = web_address

        code_generator = Generator(video_names, (video_wid, video_hei))
        code_generator.run()

        self.new_act_queue = Q(2)           # new actions from webpage
        self.display_queues = {}            # image queues
        for v in video_names:
            self.display_queues[v] = Q(queue_size)

        self.log('init, video names %s' % str(video_names))


    def run(self):
        ip, port = self.web_address.split(':')
        from web.web_server import WebServer
        wserver = WebServer(data_queues=self.display_queues, 
                            msg_queue=self.new_act_queue, 
                            ip=ip, port=int(port))
        self.log('web server started')
        wserver.run()
        self.log('webserver ended')


    def get_new_act(self):
        ''' Read new act from webpage 
        '''
        return self.new_act_queue.read()


    def write_img_queue(self, im, cam_id):
        ''' Write image to the display queue
        '''
        self.display_queues[cam_id].write(im)


    def log(self, s):
        logging.debug('[Web] %s' % s)
