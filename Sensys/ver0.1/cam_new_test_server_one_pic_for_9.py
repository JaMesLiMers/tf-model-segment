#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import os
import cv2
import numpy as np
import os.path
import net_max3
import weights_loader
import time
import json
import socket
from socketIO_client_nexus import SocketIO, LoggingNamespace
import base64
import threading

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
  if thresholded_predictions == []:
      pass
  else:
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

def postprocessing(predictions,input_image,score_threshold,iou_threshold,input_height,input_width):

  input_image = cv2.resize(input_image,(input_height, input_width), interpolation = cv2.INTER_CUBIC)

  if predictions == []:
      pass
  else:
      n_classes = 20
      n_grid_cells = 13
      n_b_boxes = 5
      n_b_box_coord = 4

      # Names and colors for each class
      classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
      colors = [(254.0, 254.0, 254), (239.88888888888889, 211.66666666666669, 127),
                  (225.77777777777777, 169.33333333333334, 0), (211.66666666666669, 127.0, 254),
                  (197.55555555555557, 84.66666666666667, 127), (183.44444444444443, 42.33333333333332, 0),
                  (169.33333333333334, 0.0, 254), (155.22222222222223, -42.33333333333335, 127),
                  (141.11111111111111, -84.66666666666664, 0), (127.0, 254.0, 254),
                  (112.88888888888889, 211.66666666666669, 127), (98.77777777777777, 169.33333333333334, 0),
                  (84.66666666666667, 127.0, 254), (70.55555555555556, 84.66666666666667, 127),
                  (56.44444444444444, 42.33333333333332, 0), (42.33333333333332, 0.0, 254),
                  (28.222222222222236, -42.33333333333335, 127), (14.111111111111118, -84.66666666666664, 0),
                  (0.0, 254.0, 254), (-14.111111111111118, 211.66666666666669, 127)]

      # Pre-computed YOLOv2 shapes of the k=5 B-Boxes
      anchors = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]

      thresholded_predictions = []
      print('Thresholding on (Objectness score)*(Best class score) with threshold = {}'.format(score_threshold))

      # IMPORTANT: reshape to have shape = [ 13 x 13 x (5 B-Boxes) x (4 Coords + 1 Obj score + 20 Class scores ) ]
      # From now on the predictions are ORDERED and can be extracted in a simple way!
      # We have 13x13 grid cells, each cell has 5 B-Boxes, each B-Box have 25 channels with 4 coords, 1 Obj score , 20 Class scores
      # E.g. predictions[row, col, b, :4] will return the 4 coords of the "b" B-Box which is in the [row,col] grid cell
      predictions = np.reshape(predictions,(13,13,5,25))

      # IMPORTANT: Compute the coordinates and score of the B-Boxes by considering the parametrization of YOLOv2
      for row in range(n_grid_cells):
        for col in range(n_grid_cells):
          for b in range(n_b_boxes):

            tx, ty, tw, th, tc = predictions[row, col, b, :5]

            # IMPORTANT: (416 img size) / (13 grid cells) = 32!
            # YOLOv2 predicts parametrized coordinates that must be converted to full size
            # final_coordinates = parametrized_coordinates * 32.0 ( You can see other EQUIVALENT ways to do this...)
            center_x = (float(col) + sigmoid(tx)) * 32.0
            center_y = (float(row) + sigmoid(ty)) * 32.0

            roi_w = np.exp(tw) * anchors[2*b + 0] * 32.0
            roi_h = np.exp(th) * anchors[2*b + 1] * 32.0

            final_confidence = sigmoid(tc)

            # Find best class
            class_predictions = predictions[row, col, b, 5:]
            class_predictions = softmax(class_predictions)

            class_predictions = tuple(class_predictions)
            best_class = class_predictions.index(max(class_predictions))
            best_class_score = class_predictions[best_class]

            # Compute the final coordinates on both axes
            left   = int(center_x - (roi_w/2.))
            right  = int(center_x + (roi_w/2.))
            top    = int(center_y - (roi_h/2.))
            bottom = int(center_y + (roi_h/2.))

            if( (final_confidence * best_class_score) > score_threshold):
              thresholded_predictions.append([[left,top,right,bottom],final_confidence * best_class_score,classes[best_class]])

      # Sort the B-boxes by their final score
      thresholded_predictions.sort(key=lambda tup: tup[1],reverse=True)

      print('Printing {} B-boxes survived after score thresholding:'.format(len(thresholded_predictions)))
      for i in range(len(thresholded_predictions)):
        print('B-Box {} : {}'.format(i+1,thresholded_predictions[i]))

      # Non maximal suppression
      print('Non maximal suppression with iou threshold = {}'.format(iou_threshold))
      nms_predictions = non_maximal_suppression(thresholded_predictions,iou_threshold)

      # Print survived b-boxes
      print('Printing the {} B-Boxes survived after non maximal suppression:'.format(len(nms_predictions)))
      for i in range(len(nms_predictions)):
        print('B-Box {} : {}'.format(i+1,nms_predictions[i]))

      # Draw final B-Boxes and label on input image
      for i in range(len(nms_predictions)):

          color = colors[classes.index(nms_predictions[i][2])]
          best_class_name = nms_predictions[i][2]

          # Put a class rectangle with B-Box coordinates and a class label on the image
          input_image = cv2.rectangle(input_image,(nms_predictions[i][0][0],nms_predictions[i][0][1]),(nms_predictions[i][0][2],nms_predictions[i][0][3]),color)
          cv2.putText(input_image,best_class_name,(int((nms_predictions[i][0][0]+nms_predictions[i][0][2])/2),int((nms_predictions[i][0][1]+nms_predictions[i][0][3])/2)),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)

  return input_image

def inference(sess,pre_predictions, cut_point):
    return sess.run(net_max3.o9, feed_dict={eval("net_max3.o{}".format(cut_point)): pre_predictions})
    #return sess.run(o9,feed_dict={eval("o{}".format(cut_point)):pre_predictions})

# IMPORTANT: Weights order in the binary file is [ 'biases','gamma','moving_mean','moving_variance','kernel']
# IMPORTANT: biases ARE NOT the usual biases to add after the conv2d! They refer to the betas (offsets) in the Batch Normalization!
# IMPORTANT: the biases added after the conv2d are set to zero!
# IMPORTANT: to use the weights they actually need to be de-normalized because of the Batch Normalization! ( see later )

# --------------------多线程发送------------
class myThread(threading.Thread):
    def __init__(self, threadID, socketIO, name, frame, fps, cut_point):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.socketIO = socketIO
        self.name = name
        self.frame = frame
        self.fps = fps
        self.cut_point = cut_point
        # print(self.threadID,self.socketIO,self.name,self.frame,self.fps)

    def run(self):
        print("Start Emitting：" + self.name)
        # ------send-------
        emission = {}
        retrval, output_img = cv2.imencode('.jpg', self.frame)
        output_img = base64.b64encode(output_img)
        output_img = str(output_img)
        emission['frame'] = output_img[2:-1]
        emission['fps'] = self.fps
        emission['cut_point'] = self.cut_point
        self.socketIO.emit('img', emission)

        # -----------------
        print("End Emitting：" + self.name)

    # --------------------多线程发送------------
# -----------------连接目标服务器-------------------------------
def on_connect():
    print('connect')

def on_disconnect():
    print('disconnect')

def on_reconnect():
    print('reconnect')

# -------------------------------------------------------------

def main(_):
    counter = 0
    # Definition of the paths
    weights_path = 'yolov2-tiny-voc.weights'
    input_img_path = 'test_zzh_1.jpg'
    output_image_path = 'output/zzh_out.jpg'

    # If you do not have the checkpoint yet keep it like this! When you will run test.py for the first time it will be created automatically
    ckpt_folder_path = './ckpt/'

    # Definition of the parameters
    input_height = 416
    input_width = 416
    score_threshold = 0.1
    iou_threshold = 0.1

    # Definition of the session
    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    # Check for an existing checkpoint and load the weights (if it exists) or do it from binary file
    print('Looking for a checkpoint...')
    saver = tf.train.Saver()
    _ = weights_loader.load(sess, weights_path, ckpt_folder_path, saver)

    # ------------------------预定义和加载数据-----------------------
    # -----------------连接目标服务器-------------------------------
    socketIO = SocketIO('192.168.1.105', 3333, LoggingNamespace)
    socketIO.on('connection', on_connect)
    socketIO.on('disconnect', on_disconnect)
    socketIO.on('reconnect', on_reconnect)
    # -------------------------------------------------------------
    time_1 = 0
    time_2 = 0

    host = '127.0.0.1'
    port = 12345
    buffsize = 65535

    ADDR = (host, port)

    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.bind(ADDR)
    soc.listen(2)

    print('Wait for connection ...')
    soc_client, addr = soc.accept()
    print("Connection from :", addr)

    # 进行长连接不断接受信号， 每次data=的时候都会刷新所有接收到的值
    while True:
        data = ""
        data = soc_client.recv(buffsize).decode()
        while True:
            # data为初始化值是系统等待， 以防while占用太大的运算量
            if data == "":
                time.sleep(0.1)
            else:
                break
        # 服务器端 send 各个命令执行的操作
        if data == "time_start":
            # 用来记录总时间 （未用到）
            start_time = time.time()
            print(start_time)
            soc_client.send("done".encode())

        elif data == "time_stop":
            # 结束记录总时间 （未用到）
            stop_time = time.time()
            total_time = stop_time - start_time
            print(total_time)
            soc_client.send("done".encode())

        elif data == "finish":
            # 用来结束连接（未用到）
            soc_client.close()
            break

        elif data == "time_1":
            # 用来接受time_1信息（初始化时间长度）
            soc_client.send("recive".encode())
            time_1 = soc_client.recv(buffsize).decode()
            time_1 = float(time_1)
            soc_client.send("done".encode())

        elif data == "time_2":
            # 用来接受time_2信息 （处理的时间长度）
            soc_client.send("recive".encode())
            time_2 = soc_client.recv(buffsize).decode()
            time_2 = float(time_2)
            soc_client.send("done".encode())

        elif data == "send_original_image":
            # 用来接受输出的原始图像 （记录下下载图像的时间为 recive_time）
            soc_client.send("recive".encode())
            recive_time = time.time()
            data_batches = ""
            new_data = ""

            # 开始下载
            print("downloading...")
            while True:
                new_data = soc_client.recv(buffsize).decode()

                # 结束时的处理
                if ((new_data[-1] == '%')):
                    data_batches = data_batches + new_data[:-1]
                    break
                # 循环体
                data_batches = data_batches + new_data

            print("processing...")
            # 加载下载的原图像解析为np array（注意imshow输出时要转化成np.uint8格式否则会黑屏）（<-坑）
            data_text = json.loads(data_batches)
            input_image = np.array(data_text)

            # 下载时间记录 （下载 + 解析）
            recive_time = time.time() - recive_time
            print("total time: {}".format(recive_time))
            soc_client.send("done".encode())


        elif data == "send_weight_cutpoint":
            # 下载 weight 和 cut point
            soc_client.send("recive".encode())
            # 记录时间 p4
            point_4 = time.time()
            data_batches = ""
            new_data = ""
            cut_point = ""

            # 开始下载
            print("downloading...")
            while True:
                new_data = soc_client.recv(buffsize).decode()
                # 结束时的处理 （cut point 作为倒数第二个值被接受在新变量中）
                if ((new_data[-1] == '%')):
                    data_batches = data_batches + new_data[:-2]
                    cut_point = new_data[-2:-1]
                    # print(cut_point)
                    break
                # 循环体
                data_batches = data_batches + new_data

            # 记录时间 p5 （time_3 为下载weight的时间）
            point_5 = time.time()
            time_3 = point_5 - point_4
            print("downloading time: {}".format(time_3))

            print("processing...")
            # 加载和解析数据
            data_text = json.loads(data_batches)
            predictions = np.array(data_text)
            cut_point = int(cut_point)
            predictions = inference(sess, predictions, cut_point)

            # 记录时间 p6 （time_4 为解析数据的时间）
            point_6 = time.time()
            time_4 = point_6 - point_5
            print("processing time: {}".format(time_4))

            print('Postprocessinasg...')
            # out_put images
            output_image = postprocessing(predictions, input_image, score_threshold, iou_threshold, input_height,
                                          input_width)
            # 记录时间 p7
            # time_5 为后处理的时间
            # 计算出后端的运行时间，中间的下载时间， 前端的预处理时间（不包括初始化时间），算出fps
            point_7 = time.time()
            time_5 = point_7 - point_6
            time_backend = time_4 + time_5
            time_downloading = time_3 + recive_time
            time_frontend = time_2
            time_total = time_backend + time_frontend + time_downloading
            Fps = 1./time_total
            fps = "fps = {}".format(Fps)

            # 将图像格式转换成uint8 否则黑屏
            # 输出fps， 输出处理后的图像
            output_image = np.uint8(output_image)
            cv2.putText(output_image, str(fps), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
            #cv2.imshow("hello", output_image)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break

            # 打印出所有可能用到的时间
            counter += 1
            print("time_load_model = {}".format(time_1))
            print("time_preprocess = {}".format(time_2))
            print("time_send_original_image = {}".format(recive_time))
            print("time_downloading = {}".format(time_downloading))
            print("time_load_jsons = {}".format(time_4))
            print("time_postprocess = {}".format(time_5))
            print("time_backend = {}".format(time_backend))
            print("time_frontend = {}".format(time_frontend))
            print("Fps = {}".format(Fps))

            #--------------------多线程发送------------
            # Create a thread
            thread_socketIO = myThread(1,socketIO, "Frame-"+str(counter), output_image, Fps, cut_point)

            # Start a thread
            thread_socketIO.start()
            thread_socketIO.join()
            if counter == 9:
                socketIO.emit('complete')
            # --------------------多线程发送------------
            # 返回确认done

            soc_client.send("done".encode())
            # 下一个循环待命
            pass
        else:
            pass


    #print(point_1,point_2,point_3,point_4,point_5,type(output_image))

if __name__ == '__main__':
     tf.app.run(main=main)

#######################################################################################################################################
