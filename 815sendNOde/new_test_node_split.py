# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import numpy as np
import net
import weights_loader
import cv2
import time
import warnings
import json
import socket
import sys

warnings.filterwarnings('ignore')


def sigmoid(x):
  return 1. / (1. + np.exp(-x))



def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out



def iou(boxA,boxB):
  # boxA = boxB = [x1,y1,x2,y2]

  # Determine the coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  # Compute the area of intersection
  intersection_area = (xB - xA + 1) * (yB - yA + 1)

  # Compute the area of both rectangles
  boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

  # Compute the IOU
  iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

  return iou



def non_maximal_suppression(thresholded_predictions,iou_threshold):

  nms_predictions = []

  # Add the best B-Box because it will never be deleted
  nms_predictions.append(thresholded_predictions[0])

  # For each B-Box (starting from the 2nd) check its iou with the higher score B-Boxes
  # thresholded_predictions[i][0] = [x1,y1,x2,y2]
  i = 1
  while i < len(thresholded_predictions):
    n_boxes_to_check = len(nms_predictions)
    #print('N boxes to check = {}'.format(n_boxes_to_check))
    to_delete = False

    j = 0
    while j < n_boxes_to_check:
        curr_iou = iou(thresholded_predictions[i][0],nms_predictions[j][0])
        if(curr_iou > iou_threshold ):
            to_delete = True
        #print('Checking box {} vs {}: IOU = {} , To delete = {}'.format(thresholded_predictions[i][0],nms_predictions[j][0],curr_iou,to_delete))
        j = j+1

    if to_delete == False:
        nms_predictions.append(thresholded_predictions[i])
    i = i+1

  return nms_predictions



def preprocessing(input_img_path,input_height,input_width):

  input_image = cv2.imread(input_img_path)

  # Resize the image and convert to array of float32
  resized_image = cv2.resize(input_image,(input_height, input_width), interpolation = cv2.INTER_CUBIC)
  image_data = np.array(resized_image, dtype='f')

  # Normalization [0,255] -> [0,1]
  image_data /= 255.

  # BGR -> RGB? The results do not change much
  # copied_image = image_data
  #image_data[:,:,2] = copied_image[:,:,0]
  #image_data[:,:,0] = copied_image[:,:,2]

  # Add the dimension relative to the batch size needed for the input placeholder "x"
  image_array = np.expand_dims(image_data, 0)  # Add batch dimension

  return image_array


def inference(sess,preprocessed_image,cut_point):
  # Forward pass of the preprocessed image into the network defined in the net.py file
  predictions = sess.run(eval("net.o{}".format(cut_point)),feed_dict={net.x:preprocessed_image})
  #print(predictions,type(predictions))
  return predictions


def connect_to_server(HOST, PORT):

    # 创建连接
    ADDR = (HOST, PORT)
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    soc.connect(ADDR)

    return soc

def send_time(time_name, time_num, soc, buffsize):
    soc.send("{}".format(time_name).encode())
    recive = soc.recv(buffsize).decode()
    if recive != "recive":
        time.sleep(0.1)
    elif recive == "recive":
        soc.send("{}".format(time_num).encode())
        done = soc.recv(buffsize).decode()
        if done != "done":
            time.sleep(0.1)
        else:
            print("send {}".format(time_name))
            pass


def send_weight_and_time(sess, preprocessed_image, cut_point, soc, buffsize):

    # predict前几层
    predictions = inference(sess, preprocessed_image, cut_point)
    print("predicting...")
    # 开始传输
    soc.send("send_weight_cutpoint".encode())

    recive = soc.recv(buffsize).decode()
    if recive != "recive":
        time.sleep(0.1)
    elif recive == "recive":
        # json 压缩部分
        print("compiling data batches")
        # 记录时间1
        time_conpile_1 = time.time()
        # 开始dump
        x = predictions.tolist()
        json_x = json.dumps(x)
        print(sys.getsizeof(json_x))
        json_x = json_x.encode()
        # 记录时间2
        time_compile_2 = time.time()
        time_compile = time_compile_2 - time_conpile_1
        print(time_compile)

        # 开始传输
        print("Sending data batches")
        while True:
            if (len(json_x) < buffsize):
                soc.send(json_x[:buffsize])
                print("send weight")
                # tctimeClient.close()
                break
            soc.send(json_x[:buffsize])
            json_x = json_x[buffsize:]

        # 传输后传输 cut point
        soc.send("{}%".format(cut_point).encode())

        done = soc.recv(buffsize).decode()
        if done != "done":
            time.sleep(0.1)
        else:
            print("all send")
            pass
    else:
        raise Exception("Nothing send back")


### MAIN ##############################################################################################################

def main(_):

    point_0 = time.time()
    # Definition of the paths
    weights_path = 'yolov2-tiny-voc.weights'
    input_img_path = 'test_zzh.jpg'
    output_image_path = 'output/zzh_out.jpg'

    # If you do not have the checkpoint yet keep it like this! When you will run test.py for the first time it will be created automatically
    ckpt_folder_path = './ckpt/'

    # Definition of the parameters
    input_height = 416
    input_width = 416
    cut_point = 3
    # Definition of the session
    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    # Check for an existing checkpoint and load the weights (if it exists) or do it from binary file
    print('Looking for a checkpoint...')
    saver = tf.train.Saver()
    _ = weights_loader.load(sess,weights_path,ckpt_folder_path,saver)

    # 时间 1 加载参数时间
    time_1 = time.time() - point_0

    # Preprocess the input image
    print('Preprocessing...')
    preprocessed_image = preprocessing(input_img_path,input_height,input_width)

    # 时间2 预处理时间
    time_2 = time.time() - point_0 - time_1

    # 建立连接
    HOST = '192.168.1.100'
    PORT = 12345
    buffsize = 65535
    soc = connect_to_server(HOST, PORT)

    send_time("time_1", time_1, soc, buffsize)
    send_time("time_2", time_2, soc, buffsize)

    print('Sending weight')
    for i in range(9):
        send_weight_and_time(sess, preprocessed_image, i+1, soc, buffsize)

if __name__ == '__main__':
     tf.app.run(main=main)

#######################################################################################################################
