from __future__ import absolute_import, division, print_function

"""
For debugging use, serves as a server: 
- move this file to the home folder of Caesar 
- run $ python main_dummy.py [local_address] [local_port] [server_addr_and_port]
        # if [server_addr_and_port] is input, will upload data to there
"""

import sys 
import socket
import os
from time import time, sleep
from multiprocessing import Process
from threading import Thread
from network.socket_server import NetServer


QUEUE_SIZE = 64 
LOCAL_NAME = 'dummy'
HEADER = b'\x00\x00CAESAR\x00\x00'


def main(running, address, port, server_addr):
    UPLOAD_DATA = True if server_addr else False 
    if not server_addr:
        server_addr = "localhost:0"

    server = NetServer(name=LOCAL_NAME,
                        address=address,
                        port=port,
                        buffer_size=QUEUE_SIZE)
    server_proc = Process(target=server.run)
    server_proc.start()

    server_ip, server_port = server_addr.split(':')
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if UPLOAD_DATA:
        while True:
            try:
                client_socket.connect((server_ip, int(server_port)))
                break 
            except ConnectionRefusedError:
                print("cannot connect to server, reconnecting...")
                sleep(10)

    print('Dummy starts!')
    while running[0]:
        pkt = server.data_queue.read()
        if pkt is None:
            sleep(0.01)
            continue

        print('pkt: %d' % pkt.frame_id)

        if not UPLOAD_DATA:
            continue 

        need_reconnect = False     
        d = pkt.encode()

        while True:
            try:
                client_socket.send(HEADER + d)
                break 
            except socket.error:
                print('Cannot send pkt')
                
            sleep(1)
            client_socket.close()
            print('closed the socket, try reconnecting...')
            
            while True:
                try:
                    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    # client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    client_socket.connect((server_ip, int(server_port)))
                    break 
                except ConnectionRefusedError:
                    print("cannot connect to server, reconnecting...")
                    sleep(10)

    client_socket.close()
    server.stop()
    print('dummy finished')


if __name__ == '__main__':
    running = [True]
    address=sys.argv[1]
    port = int(sys.argv[2])
    server_addr = sys.argv[3] if len(sys.argv) > 3 else ''

    th = Thread(target=main, args=(running, address, port, server_addr))
    th.start()
    while True:
        try:
            sleep(10)
        except (KeyboardInterrupt, SystemExit):
            running[0] = False
            break

    print('done')

