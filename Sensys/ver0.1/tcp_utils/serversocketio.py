from flask import Flask, render_template
from flask_socketio import SocketIO
import base64
import cv2


app = Flask(__name__)
socketio = SocketIO(app)

if __name__ == '__main__':
    socketio.run(app)


@socketio.on('message')
def handle_message(message):
    print('received message: ' + message)






output_img = cv2.imread(input_img_path)


def on_connect():
    print('connect')

def on_disconnect():
    print('disconnect')

def on_reconnect():
    print('reconnect')




socketIO = SocketIO('192.168.1.104', 3334, LoggingNamespace)

socketIO.on('connection', on_connect)
socketIO.on('disconnect', on_disconnect)
socketIO.on('reconnect', on_reconnect)

retrval, output_img = cv2.imencode('.jpg', output_img)
output_img = base64.b64encode(output_img)
output_img = str(output_img)
socketIO.emit('x', output_img)

socketIO.wait(50)