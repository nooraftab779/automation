import socket
import psutil
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname) 
import platform
import time

host = '192.168.88.158'
port = 5001
s = socket.socket()
s.connect((host, port))
while True:
    a = [platform.node() , platform.processor() , str(round((((psutil.virtual_memory().total)/1024)/1024)/1024)),psutil.cpu_percent()]
    s.send(bytes(str(a),'utf-8'))
    time.sleep(10)
