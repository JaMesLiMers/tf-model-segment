from socketIO_client_nexus import SocketIO, LoggingNamespace
import threading
import time
import psutil


def on_connect():
    print('connect')


def on_disconnect():
    print('disconnect')


def on_reconnect():
    print('reconnect')

def read_test(*args):
    print("test is :", args)


socketIO = SocketIO('192.168.1.105', 3333, LoggingNamespace)
socketIO.on('connection', on_connect)
socketIO.on('disconnect', on_disconnect)
socketIO.on('reconnect', on_reconnect)
socketIO.on('test', read_test)


# ---------------
