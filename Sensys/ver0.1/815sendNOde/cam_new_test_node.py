#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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
import argparse
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.description='test one picture'
parser.add_argument("-p", "--cut_point", help='Select cut point', type=int, default=9, required=False)
args = parser.parse_args()

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

def enable_video_capture(input_cam_num):
    """启用网络摄像头， 返回摄像头的接口"""
    video_capture = cv2.VideoCapture(input_cam_num)
    return video_capture

def preprocessing(video_capture, input_height,input_width):
    """调用已经开启的摄像头接口，需要设置图像高度和宽度，返回一个归一化后的图像矩阵和一个原始图像矩阵"""
    _, input_image = video_capture.read()

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

    return image_array, resized_image


def inference(sess,preprocessed_image,cut_point):
    """前向传播"""
    # Forward pass of the preprocessed image into the network defined in the net.py file
    predictions = sess.run(eval("net.o{}".format(cut_point)),feed_dict={net.x:preprocessed_image})
    #print(predictions,type(predictions))
    return predictions


def connect_to_server(HOST, PORT):
    """用来连接到目标服务器， 返回一个连接接口"""
    # 创建连接
    ADDR = (HOST, PORT)
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    soc.connect(ADDR)

    return soc


def send_original_image(original_image, soc, buffsize):
    """调用连接接口向服务器发送图像（参照服务器端看）"""
    print("send original image...")
    soc.send("send_original_image".encode())
    # 如果收到了
    recive = soc.recv(buffsize).decode()

    if recive == "recive":
        # json 压缩部分
        print("compiling image batches")
        # 记录时间1
        time_compile_1 = time.time()
        # 开始dump
        y = original_image.tolist()
        json_y = json.dumps(y)
        json_y = json_y.encode()
        # 记录时间2
        time_compile_2 = time.time()
        time_compile = time_compile_2 - time_compile_1
        print("compile time : {}".format(time_compile))

        # 开始传输
        print("Sending data batches")
        while True:
            if (len(json_y) < buffsize):
                soc.send(json_y[:buffsize])
                print("send batch")
                # tctimeClient.close()
                break
            soc.send(json_y[:buffsize])
            json_y = json_y[buffsize:]

        # 传输后传输 结束符号
        soc.send("%".encode())

        done = soc.recv(buffsize).decode()
        if done == "done":
            print("all send")
        else:
            raise Exception("Not send back 'done'")
            pass
    else:
        raise Exception("Not send back 'receive'")



def send_time(time_name, time_num, soc, buffsize):
    """调用连接接口向服务器发送需要的时间"""
    soc.send("{}".format(time_name).encode())
    recive = soc.recv(buffsize).decode()

    if recive == "recive":
        soc.send("{}".format(time_num).encode())
        done = soc.recv(buffsize).decode()
        if done == "done":
            print("all send")
        else:
            raise Exception("Not send back 'done'")
            pass
    else:
        raise Exception("Not send back 'receive'")


def send_weight_and_time(sess, preprocessed_image, cut_point, soc, buffsize):
    """调用接口像服务器发送前向传播后的weight和断点信息"""
    # predict前几层
    print("predicting...")
    predictions = inference(sess, preprocessed_image, cut_point)
    # 开始传输
    soc.send("send_weight_cutpoint".encode())

    recive = soc.recv(buffsize).decode()
    if recive == "recive":
        # json 压缩部分
        print("compiling data batches")
        # 记录时间1
        time_compile_1 = time.time()
        # 开始dump
        x = predictions.tolist()
        json_x = json.dumps(x)
        json_x = json_x.encode()
        # 记录时间2
        time_compile_2 = time.time()
        time_compile = time_compile_2 - time_compile_1
        print("compile time : {}".format(time_compile))

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
        if done == "done":
            print("all send")
        else:
            raise Exception("Not send back 'done'")
            pass
    else:
        raise Exception("Not send back 'receive'")


### MAIN ##############################################################################################################

def main(_):

    # 记录开始时间
    point_0 = time.time()
    # Definition of the paths
    weights_path = 'yolov2-tiny-voc.weights'
    input_img_path = 'test_zzh.jpg'
    output_image_path = 'output/zzh_out.jpg'

    # If you do not have the checkpoint yet keep it like this! When you will run test.py for the first time it will be created automatically
    ckpt_folder_path = './ckpt/'

    # Definition of the parameters
    # 宽高，摄像头编号，断点位置
    input_height = 416
    input_width = 416
    input_cam_num = 0
    cut_point = args.cut_point
    # Definition of the session
    sess = tf.InteractiveSession()

    # 全局进行初始化
    tf.global_variables_initializer().run()

    # Check for an existing checkpoint and load the weights (if it exists) or do it from binary file
    print('Looking for a checkpoint...')
    saver = tf.train.Saver()
    _ = weights_loader.load(sess,weights_path,ckpt_folder_path,saver)

    # time_1 为初始化的时间， 只需要用一次，记录下
    time_1 = time.time() - point_0

    # 建立连接
    HOST = '127.0.0.1'
    PORT = 12345
    buffsize = 65535
    soc = connect_to_server(HOST, PORT)

    video_capture = enable_video_capture(input_cam_num)
    time.sleep(1)
    counter = 0
    while True:
        counter += 1
        print("frame {}".format(counter))
        # time_2 开始计时
        time_2_1 = time.time()
        # Preprocess the input image
        print('Preprocessing...')
        preprocessed_image, original_image = preprocessing(video_capture,input_height,input_width)


        # time_2 为预处理时间
        time_2 = time.time() - time_2_1

        # 发送 前面收集到的两个时间
        send_time("time_1", time_1, soc, buffsize)
        send_time("time_2", time_2, soc, buffsize)

        # 发送 原图
        send_original_image(original_image, soc, buffsize)

        # 发送 weight 数据和 cut_point 信息
        send_weight_and_time(sess, preprocessed_image, cut_point, soc, buffsize)

        # 打印出图像（不需要就注释掉）
        #cv2.imshow('frame', original_image)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
    # 使用完成后释放相机并关闭所有imshow窗口（目前直接断开的，没有用到）
    #cap.release()
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    tf.app.run(main=main)

#######################################################################################################################
