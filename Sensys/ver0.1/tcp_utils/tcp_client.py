import json
import socket
import time
import base64
import cv2

weights_path = 'yolov2-tiny-voc.weights'
input_img_path = 'test_zzh.jpg'
output_image_path = 'output/zzh_out.jpg'

HOST = '192.168.1.104'

PORT = 12345
buffsize = 65535

ADDR = (HOST, PORT)

predictions = cv2.imread(input_img_path)
soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
soc.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
soc.connect(ADDR)

soc.send("time_start".encode())

time.sleep(2)

# 开始传输

soc.send("send_weight_cutpoint".encode())

time_1 = time.time()
# json 压缩部分
print("compiling data batches")
retrval, x = cv2.imencode('.jpg', predictions)
json_x = base64.b64encode(x)
json_x = str(json_x)
print(type(json_x), json_x)

time_1 = time.time() - time_1
# 开始传输
print("Sending data batches")
while True:
    if (len(json_x) < buffsize):
        soc.send(json_x[:buffsize].encode())
        soc.send("%".encode())
        print("send weight")
        # soc.close()
        break
    soc.send(json_x[:buffsize].encode())
    json_x = json_x[buffsize:]



time.sleep(1)



