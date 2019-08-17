from time import time, sleep
import grpc
import pickle
import sys
import logging 
import network.cas_proto_pb2 as cas_pb2
import network.cas_proto_pb2_grpc as cas_pb2_grpc
from network.utils import Q

import cv2
from network.data_packet import DataPkt


class NetClient:
    def __init__(self, client_name, server_addr, buffer_size):
        '''
        Args:
        - clien_name: identifier of the camera (same as the one in topo)
        - server_addr: ip:port in str
        - buffer_size: size of request_queue, and response queue
        '''
        channel = grpc.insecure_channel(server_addr)
        self.stub = cas_pb2_grpc.UploaderStub(channel)
        self.client_name = client_name
        self.request_queue = Q(buffer_size)
        self.response_queue = Q(buffer_size)
        self.log('init, connect to %s' % server_addr)


    def send_data(self, pkt):
        ''' Public: send data to rpc queue
        '''
        self.request_queue.write(pkt)

        if self.request_queue.full():
            self.log('send queue full!')
            return False

        return True


    def send_data_wait(self, pkt):
        ''' Public: Keep sending the pkt until succeed
        '''
        while not self.send_data(pkt):
            sleep(0.1)


    def read_response(self):
        ''' Public: read response queue
        '''
        return self.response_queue.read()


    def send_rpc(self, data):
        ret = self.stub.Upload(cas_pb2.UploadRequest( name=self.client_name,
                                                        data=data))
        if ret:
            return ret.message
        return ''


    def run(self):
        ''' Keeps running until get empty input_pkt
        '''
        self.log('running!')
        while True:
            input_pkt = self.request_queue.read()
            if input_pkt is None:
                sleep(0.01)
                continue

            reply = self.send_rpc(input_pkt.encode())

            if reply:
                self.response_queue.write(reply)

        self.log('done')


    def log(self, s):
        logging.debug('[Client] %s' % s)



if __name__ == '__main__':
    client = NetClient('a', 'localhost:50051', 32)
    pkt = DataPkt(img=cv2.imread(sys.argv[1]))
    for i in range(100):
        cur = time()
        ret = client.send_rpc(pkt.encode())
        print('time: %f' % (time() - cur))
        # sleep(0.4)
