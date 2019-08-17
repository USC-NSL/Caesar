import socket
import cv2 
import sys 
import logging 
import pickle 
from struct import pack
from time import time, sleep 
from network.utils import Q
from network.data_packet import DataPkt


HEADER = b'\x00\x00CAESAR\x00\x00'


class NetClient:
    def __init__(self, client_name, server_addr, buffer_size):
        self.client_name = client_name
        self.server_ip = server_addr.split(':')[0]
        self.server_port = int(server_addr.split(':')[1])
        self.request_queue = Q(buffer_size)
        self.frame_cnt = 0 

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.running = True
        self.log('init')

    def log(self, s):
        logging.info('[NetClient]: %s' % (s))

    def send_data(self, pkt):
        ''' 
        Public: send data to rpc queue
        '''
        self.request_queue.write(pkt)
        self.frame_cnt += 1

        if self.frame_cnt % 20 == 0:
            self.log('Sending pkt %d' % self.frame_cnt)
            
        if self.request_queue.full():
            if self.frame_cnt % 10 == 0:
                self.log('send queue full!')
            return False

        return True

    def run(self):
        ''' 
        Keeps running until get empty input_pkt
        '''
        while self.running:
            try:
                self.socket.connect((self.server_ip, self.server_port))
                break 
            except ConnectionRefusedError:
                print("cannot connect to server, reconnecting...")
                sleep(10)
        self.log('connected to %s!' % self.server_ip)

        while self.running:
            input_pkt = self.request_queue.read()
            if input_pkt is None:
                sleep(0.01)
                continue

            d = input_pkt.encode()

            while True:
                try:
                    self.socket.send(HEADER + d)
                    break 
                except socket.error:
                    print('Cannot send pkt')
                    
                sleep(1)
                self.socket.close()
                print('closed the socket, try reconnecting...')
                
                while True:
                    try:
                        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        self.socket.connect((self.server_ip, self.server_port))
                        break 
                    except ConnectionRefusedError:
                        print("cannot connect to server, reconnecting...")
                        sleep(10)

        self.log('done')

    def close(self):
        self.running = False 
        self.socket.close()
        self.log('Connection closed')
