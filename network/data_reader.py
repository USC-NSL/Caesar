import os
import cv2
import numpy as np
import sys
from time import time
import logging 


class DataReader:

    def __init__(self, video_path='', file_path='', max_frame_id=-1):
        ''' Read video and metadata file (opt) and output the data sequence

        Input:
            - Video file path
            - Optional: metadata file path (an npy file)
                    format: [{'frame_id': frame_id, xxx}]
                        xxx means you can put whatever k-v entry
        '''
        self.end_of_video = False
        if not video_path.isdigit() and not os.path.exists(video_path):
            self.log('Cannot load video: %s' % str(video_path))
            self.end_of_video = True
            return

        self.data = []
        self.frame_id = 0
        self.max_fid = max_frame_id

        src = int(video_path) if video_path.isdigit() else video_path
        self.cap = cv2.VideoCapture(src)

        self.data_ptr = 0
        if file_path:
            if os.path.exists(file_path):
                self.data = np.load(open(file_path, 'rb'), allow_pickle=True)
            else:
                self.log('Cannot load metadata: {}'.format(file_path))

        self.log('init')


    def read_frame(self):
        if self.end_of_video:
            return np.array([])

        ret, frame = self.cap.read()
        if self.frame_id == 0:
            self.log('Size {}'.format(frame.shape))

        if not ret or (self.max_fid > 0 and self.frame_id > self.max_fid):
            self.log('End of video')
            print('End of video!')
            self.end_of_video = True
            return np.array([])

        self.frame_id += 1
        return frame


    def get_data(self):
        '''
        Return:
        - img
        - frame_id
        - meta as a list
        '''
        if self.end_of_video or self.data_ptr >= len(self.data):
            if len(self.data):                  # read all metadata
                self.end_of_video = True
            return self.read_frame(), self.frame_id, []

        fid = self.data[self.data_ptr]['frame_id']
        frame = None
        while not self.end_of_video and self.frame_id <= fid:
            frame = self.read_frame()

        self.data_ptr += 1
        return frame, fid, self.data[self.data_ptr - 1]['meta']


    def log(self, s):
        logging.debug('[DataReader] %s' % s)
