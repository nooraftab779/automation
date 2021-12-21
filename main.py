import socket
import threading
import queue
import tqdm
from sklearn.pipeline import make_pipeline
import glob
import os
from pathlib import Path
# from custom import supertrain
# q = queue.Queue()
# from custom import train
from automation import *
from unet import *
import time
from data import *
# from UNET.train import *
parentdir = Path(os.getcwd())
#
IP = "0.0.0.0"
PORT = 5001
ADDR = (IP, PORT)
SIZE = 4096
SEPARATOR = "<SEPARATOR>"
def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    while True:
        received = conn.recv(SIZE).decode('utf-8')
        print(received)
        if(received):
            filename, filesize = received.split(SEPARATOR)
            # remove absolute path if there is
            filename = os.path.join(os.path.join(parentdir, 'zipfolder'), os.path.basename(filename))
            # convert to integer
            filesize = int(filesize)
            progress = tqdm.tqdm(range(filesize), f"Receiving {filename}", unit="B", unit_scale=True, unit_divisor=1024)
            with open(filename, "wb") as f:
                while True:
                    bytes_read = conn.recv(SIZE)
                    if not bytes_read:
                        # nothing is received
                        # file transmitting is done
                        break
                    f.write(bytes_read)
                    progress.update(len(bytes_read))
#         t = threading.currentThread()
        print(filename)
        if "aug" in filename:
            augmentationdata()
        if "unet" in filename:
            UnetMasks()
            data()
            while os.path.exists(f"{parentdir}{os.path.sep}current training{os.path.sep}UNET.txt"):
                time.sleep(1)
                print("Waiting Untill UNET is Complted")
            while os.path.exists(f"{parentdir}{os.path.sep}current training{os.path.sep}yolo.txt"):
                time.sleep(1)
                print("Waiting Untill YOLO is Complted")
            while os.path.exists(f"{parentdir}{os.path.sep}current training{os.path.sep}MRCNN.txt"):
                time.sleep(1)
                print("Waiting Untill MRCNN is Complted")
            Hello()
        # if "mrcnn" in filename:
        #     MrCnnParsing()
        #     while os.path.exists(f"{parentdir}{os.path.sep}current training{os.path.sep}UNET.txt"):
        #         time.sleep(1)
        #         print("Waiting Untill UNET is Complted")
        #     while os.path.exists(f"{parentdir}{os.path.sep}current training{os.path.sep}Yolo.txt"):
        #         time.sleep(1)
        #         print("Waiting Untill YOLO is Complted")
        #     while os.path.exists(f"{parentdir}{os.path.sep}current training{os.path.sep}MRCNN.txt"):
        #         time.sleep(1)
        #         print("Waiting Untill MRCNN is Complted")
        #     supertrain()
        if "yolov" in filename:
            yoloannotation()
            while os.path.exists(f"{parentdir}{os.path.sep}current training{os.path.sep}UNET.txt"):
                time.sleep(1)
                print("Waiting Untill UNET is Complted")
            while os.path.exists(f"{parentdir}{os.path.sep}current training{os.path.sep}Yolo.txt"):
                time.sleep(1)
                print("Waiting Untill YOLO is Complted")
            while os.path.exists(f"{parentdir}{os.path.sep}current training{os.path.sep}MRCNN.txt"):
                time.sleep(1)
                print("Waiting Untill MRCNN is Complted")
            yolo()
        print(f"[DISCONNECTED] {addr} disconnected")
    conn.close()
def main():
    # augmentationdata()
    # yoloannotation()
    # yolo()
    # MrCnnParsing()
    # train()
    # UnetMasks()
    # data()
    print("[STARTING] Server is starting")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)
    server.listen()
    print(f"[LISTENING] Server is listening on {IP}:{PORT}.")

    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")

if __name__ == "__main__":
    main()