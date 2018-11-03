from socketIO_client_nexus import SocketIO, LoggingNamespace
import base64
import cv2




def on_connect():
    print('connect')

def on_disconnect():
    print('disconnect')

def on_reconnect():
    print('reconnect')



print("begin")


socketIO = SocketIO('192.168.1.104', 3333, LoggingNamespace)

socketIO.on('connection', on_connect)
socketIO.on('disconnect', on_disconnect)
socketIO.on('reconnect', on_reconnect)

retrval, output_img = cv2.imencode('.jpg', output_img)
output_img = base64.b64encode(output_img)
output_img = str(output_img)
socketIO.emit('x', output_img)

print("over")

socketIO.wait(1)