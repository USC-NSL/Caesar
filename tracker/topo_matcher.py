from collections import defaultdict 
import os 
import logging 
from server.action_spatial import overlap

class TopoMatcher:
    def __init__(self, topo_file, img_shape):
        ''' This class decides if a tube should be matched to tubes in a camera
            based on their enter/exit positions 

        Args:
        - topo_file: path to the file that specifies camera-topo
            Each line: cam_id,x0,y0,x1,y1 : cam_id,x0,y0,x1,y1
            The left half is the entrance area in current camera
            The right half is the exit area in previous camera (x y in ratios)
        - img_shape: (w, h) number of pixels of the image 
        '''

        # {cam_1: cam_2: {'entry_zone': cam_1_range, 'exit_zone': cam_2_range}}
        self.topo = defaultdict(dict)
        if not os.path.exists(topo_file):
            return 

        lines = []
        with open(topo_file, 'r') as fin:
            lines = fin.readlines()

        w, h = img_shape
        for line in lines:
            line = line.strip()
            if not line or '#' in line:
                continue 
            d1, d2 = line.split(' : ')
            c1, x10, y10, x11, y11 = d1.split(', ')
            c2, x20, y20, x21, y21 = d2.split(', ')
            self.topo[c1][c2] = {'entry_zone':(
                                    int(w * float(x10)), int(h * float(y10)), 
                                    int(w * float(x11)), int(h * float(y11))),
                                'exit_zone':(
                                    int(w * float(x20)), int(h * float(y20)), 
                                    int(w * float(x21)), int(h * float(y21)))
                                }
            self.topo[c2][c1] = {'entry_zone':(
                                    int(w * float(x20)), int(h * float(y20)), 
                                    int(w * float(x21)), int(h * float(y21))),
                                'exit_zone':(
                                    int(w * float(x10)), int(h * float(y10)), 
                                    int(w * float(x11)), int(w * float(y11)))
                                }

        self.log('init: %s' % str(self.topo))


    def connected_camera(self, c1, c2):
        ''' Return if the two cameras are connected 
        '''
        return c2 in self.topo[c1]


    def can_be_matched(self, cam1, box1, cam2, box2):
        ''' Return if entry box position can be matched after the exit box
        '''
        d = self.topo[cam1][cam2]
        return overlap(box1, d['entry_zone']) and overlap(box2, d['exit_zone'])


    def log(self, s):
        logging.debug('[TOPO] %s' % s)
