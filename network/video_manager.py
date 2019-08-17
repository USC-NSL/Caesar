import cv2
import sys 
import os 
import numpy as np
import logging


class VideoIndexReader: 
    ''' Use index to read specific frame. Not in use 
    '''
    def __init__(self, src, proc_gap):
        self.src = src
        self.type = 'file'
        if self.src.isdigit():
            self.src = int(self.src)
            self.type = 'webcam'

        if self.type == 'file' and not os.path.exists(src):
            self.log('error: Cannot read video!')
            sys.exit(0)

        self.cap = cv2.VideoCapture(self.src)
        self.frame_id = 0
        self.end_of_frame = -1
        self.proc_gap = proc_gap
        self.log('init!')


    def log(self, s):
        logging.debug('[LocalVideoManager] %s' % s)


    def empty_frame(self):
        ''' return an empty frame
        '''
        return np.array([])


    def get_frame(self, target_id=-1):
        ''' Read one frame 
        Input: 
            - target_id: the client may want to read a specific frame

        Output:
            - frame: the frame itself 
            - frame_id: 
                if end of frame: frame_id < 0 
                if target_id >= cur_id: will get the target_id and return frame 
                if target_id < cur_id: will return empty frame and cur_id 
        '''
        if target_id >= 0 and target_id < self.frame_id:
            return self.empty_frame(), self.frame_id

        frame = self.empty_frame()
        while self.frame_id < target_id or self.frame_id % self.proc_gap:
            ret, frame = self.cap.read()
            self.frame_id += 1
            if not ret:
                self.log('end of video')
                return self.empty_frame(), self.end_of_frame

        ret, frame = self.cap.read()
        self.frame_id += 1
        if not ret:
            self.log('end of video')
            return self.empty_frame(), self.end_of_frame
        return frame, self.frame_id - 1



class VideoReader:
    ''' Class to read local video file or webcam 
    
    Input:
        - src: source path
    '''
    def __init__(self, src):
        self.src = src
        self.type = 'file'

        if self.src.isdigit():
            self.src = int(self.src)
            self.type = 'webcam'

        if self.type == 'file' and not os.path.exists(src):
            self.log('error: Cannot read video!')
            sys.exit(0)

        self.cap = cv2.VideoCapture(self.src)
        self.log('init!')


    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.log('end of video')
            return self.empty_frame()
        return frame


    def close(self):
        self.cap.release()
        self.log('ended')


    def log(self, s):
        logging.debug('[VideoReader] %s' % s)


    def empty_frame(self):
        ''' return an empty frame
        '''
        return np.array([])


class VideoWriter:
    ''' Class to write video to file
    
    Input:
        - fps: of the video 
        - resolution: wid x hei
        - platform: mac or linux
    '''
    def __init__(self, fname, fps, resolution, platform='Linux'):
        video_writer = cv2.VideoWriter_fourcc(*'XVID')
        self.video_saver = cv2.VideoWriter(fname, video_writer, fps, resolution)
        self.log('Create writer %s' % fname)


    def log(self, s):
        logging.debug('[VideoWriter] %s' % s)


    def close(self):
        self.video_saver.release()
        self.log('ended')


    def save_frame(self, img):
        self.video_saver.write(img)


if __name__ == '__main__':
    vr = VideoReader(sys.argv[1])
    cnt = 0 
    while True:
        f = vr.get_frame()
        if not len(f):
            break
        print('frame %d' % cnt ) 
        cnt += 1
    print('done')