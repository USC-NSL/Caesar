import socket
from threading import Thread
from time import time, sleep
import sys 
import logging 

from network.utils import Q
from network.data_packet import DataPkt


SOCKET_BUFF_SIZE = 2048
QUEUE_SIZE = 128
HEADER = b'\x00\x00CAESAR\x00\x00'


class ServerThread(Thread):
    def __init__(self, name, ip, port, sock, queue):
        Thread.__init__(self)
        self.name = name
        self.ip = ip
        self.port = port
        self.sock = sock
        self.queue = queue

        self.header = HEADER
        self.log("Server thread-" + ip + ":" + str(port))

    def log(self, s):
        logging.debug('[%s]: %s' % (self.name, s))

    def close(self):
        self.sock.close()

    def run(self):
        d = b''
        while True:
            pkt_bytes = self.sock.recv(SOCKET_BUFF_SIZE)

            if not pkt_bytes:
                self.log('Connection ended')
                self.sock.close()
                break

            d += pkt_bytes

            head = d.find(self.header)
            if head < 0:
                continue 

            next_head = d[head + 1:].find(self.header)
            if next_head < 0:
                continue

            next_head += head + 1
            
            pkt = DataPkt()
            pkt.load_from_string(d[head + len(self.header): next_head])

            d = d[next_head:]
            
            self.queue.write(pkt)

        self.close()
        self.log('Thread finished')


class NetServer(Thread):
    def __init__(self, name, address, port, buffer_size):
        Thread.__init__(self)
        self.name = name
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((address, port))

        self.data_queue = Q(buffer_size)
        self.threads = []
        self.running = True 

        self.log('Start running.')        

    def run(self):
        thread_cnt = 0
        while self.running:
            self.socket.listen(2)
            self.log("Waiting for incoming connections...")
            (conn, (ip, port)) = self.socket.accept()
            src_addr = str(ip) + ':' + str(port)
            self.log('Got connection from %s:%d' % (ip, port))
            
            newthread = ServerThread(
                'ServerThread-%d' % thread_cnt, 
                ip, 
                port, 
                conn,
                self.data_queue
            )
            newthread.setDaemon(True)
            newthread.start()
            thread_cnt += 1
            self.threads.append(newthread)

        for t in self.threads:
            t.join()
        self.log('ended')

    def read_data(self):
        return self.data_queue.read()

    def stop(self):
        self.running = False

    def log(self, s):
        logging.debug('[NetServer] %s' % s)
