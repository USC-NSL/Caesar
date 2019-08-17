"""
Usage: 
1. python3 test_connection.py -s server_ip server_port
2. python3 test_connection.py -c server_ip server_port
"""

import socket
import sys 

SOCKET_BUFF_SIZE = 2048


assert len(sys.argv) == 4, "Wrong args number!"
assert sys.argv[1] in ["-s", "-c"], "Wrong mode"!

if sys.argv[1] == "-c":
    print("As Client")
    socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket.connect((sys.argv[2], int(sys.argv[3])))
    try:
        socket.send("hello")
    except socket.error:
        print('Network error')
    _ = input("press anykey to exist")

else:
    print("As Server")
    socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    socket.bind((sys.argv[2], int(sys.argv[3])))
    socket.listen(2)
    print("Waiting for incoming connections...")
    (conn, (ip, port)) = socket.accept()
    print('Got connection from %s:%d' % (ip, port))
    pkt_bytes = sock.recv(SOCKET_BUFF_SIZE)
    print(str(pkt_bytes))
    print('Connection ended')


print('done')