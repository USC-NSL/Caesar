import os 
import numpy as np
import sys 
from time import time 
import logging 


class DataWriter:
    ''' Write metadata to file 

    Input: 
         {'frame_id': frame_id, xxx}
        xxx means you can put whatever k-v entry

    Output: 
        an npy file named in Setup
    '''

    def __init__(self, file_path='meta.npy'):
        if os.path.exists(file_path):
            self.log('WARNING: the file %s exist!' % file_path)
        
        self.file_path = file_path
        self.data = []
        self.log('init')


    def save_data(self, frame_id, meta):
        self.data.append({'frame_id': frame_id, 'meta': meta})


    def save_to_file(self):
        ''' Save the current data into npy file 
        '''

        self.log('saving data....')
        with open(self.file_path, 'wb') as fout:
            np.save(fout, self.data)
        # self.data = []
        self.log('ended')


    def save_to_txt(self):
        ''' Save to txt file instead of npy
        '''
        
        self.log('saving txt....')
        txt_file_name = self.file_path.split('.npy')[0] + '.txt'
        with open(txt_file_name, 'wb') as fout:
            for d in self.data:
                fout.write('\n%d ---------------\n' % d['frame_id'])
                fout.write(str(d['meta']) + '\n')
        self.log('ended')
        

    def log(self, s):
        logging.debug('[DataWriter] %s' % s)

