import grpc 
import pickle
from time import time, sleep
from threading import Thread
from concurrent import futures
import logging
import network.cas_proto_pb2 as cas_pb2
import network.cas_proto_pb2_grpc as cas_pb2_grpc
from network.utils import Q
from network.data_packet import DataPkt


class Uploader(cas_pb2_grpc.UploaderServicer):
    def __init__(self, data_queue, control_queues, buffer_size):
        self.data_queue = data_queue
        self.control_queues = control_queues
        self.buffer_size = buffer_size
        self.timer = time()
        print('Uploader inited')


    def Upload(self, request, context):
        '''
        Args:
        - request: the data from the client 
        '''
        pkt = DataPkt()
        pkt.load_from_string(request.data)

        cam_id = pkt.cam_id
        if cam_id not in self.control_queues:
            print('receive data from %s' % cam_id)
            self.control_queues[cam_id] = Q(self.buffer_size)

        self.data_queue.write(pkt)
        if self.data_queue.full():
            print('rpc server queue full!')

        msg = str(pkt.frame_id)
        if not self.control_queues[cam_id].empty():
            msg = self.control_queues[cam_id].read()

        return cas_pb2.UploadReply(message=msg)


class NetServer:
    def __init__(self, name, address, port, buffer_size):
        '''
        Args:
        - name: the name of local machine 
        - port: int port number
        - buffer_size: size of input data queue
        '''
        self.data_queue = Q(buffer_size)
        self.control_queues = {}
        self.name = name
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        cas_pb2_grpc.add_UploaderServicer_to_server(Uploader(data_queue=self.data_queue,
                                                            control_queues=self.control_queues,
                                                            buffer_size=buffer_size),
                                                            self.server)
        self.server.add_insecure_port('[::]:%d' % port)
        self.server.start()
        self.log('init')


    def stop(self):
        self.server.stop(0)
        self.log('ended')


    def write_response(self, client_name, msg):
        '''
        Args:
        - client_name: naem of the client that you want to send msg
        - msg: str 
        '''
        if not client_name in self.control_queues:
            self.log('Not connected to %s' % client_name)
            return

        if self.control_queues[client_name].full():
            self.log('Control queue %s is full!' % client_name)
        else:
            self.control_queues[client_name].write(msg)


    def read_data(self):
        ''' Read data from data_queue
        '''
        return self.data_queue.read()


    def log(self, s):
        logging.debug('[NetServer] %s' % s)


if __name__ == '__main__':
    server = NetServer('s',50051,32)
    _ = input('')
    server.stop()