from multiprocessing import Queue
import queue
from time import time 

''' Given a file path, extract its file name 
'''
def fname(s):
    if '/' in s:
        s = s.split('/')[-1]
    return '.'.join(s.split('.')[:-1])


# queue operations 
class Q:
    def __init__(self, qsize):
        self.q = Queue(qsize)


    def write(self, v):
        try:
            self.q.put_nowait(v)
        except queue.Full:
            pass


    def read(self):
        try:
            d = self.q.get_nowait()
            return d 
        except queue.Empty:
            return None


    def empty(self):
        return self.q.empty()


    def full(self):
        return self.q.full()
        

    def qsize(self):
        return self.q.qsize()
        

class Timer:
    ''' Class for calculating speed of a part of code 
    '''
    def __init__(self):
        self.cur_time = time()
        self.prev_fps = 0.


    def refresh(self):
        self.cur_time = time()


    def fps(self):
        cur_fps = 1. / (time() - self.cur_time)
        self.refresh()
        res = (cur_fps + self.prev_fps) / 2.
        prev_fps = cur_fps
        return res 


if __name__ == '__main__':
    '''
    im = cv2.imread(sys.argv[1])
    data = encode_img(im)
    print(len(data))
    import pickle
    print(len(pickle.dumps({'img':data,'fid':1,'box':[123,123,353]})))
    print(len(str(pickle.dumps({'img':data,'fid':1,'box':[123,123,353]}))))
    '''