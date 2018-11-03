import socket
import time
import json

host = "127.0.0.1"
port = 12345
buffsize = 65535

ADDR = (host, port)

soc = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
soc.bind(ADDR)
soc.listen(2)

weights_path = 'yolov2-tiny-voc.weights'
input_img_path = 'test_zzh_1.jpg'
output_image_path = 'output/zzh_out.jpg'

print('Wait for connection ...')
soc_client, addr = soc.accept()
print("Connection from :", addr)
while True:
    data = soc_client.recv(buffsize).decode()
    # 服务器端 send 各个命令执行的操作
    if data == "time_start":
        # 用来记录总时间
        start_time = time.time()
        print(start_time)
    elif data == "time_stop":
        stop_time = time.time()
        total_time = stop_time - start_time
        print(total_time)
        tctimeClient.close()
    elif data == "send_weight_cutpoint":
        # 记录时间 p4
        point_4 = time.time()
        data_batches = ""
        cut_point = None

        print("downloading...")
        while True:
            new_data = soc_client.recv(buffsize).decode()
            # print(len(new_data))
            # 结束时的处理
            if ((new_data[-1] == '%')):
                data_batches += new_data[:-2]
                # 接受 cutpoint部分
                cut_point = new_data[-2:-1]
                # print(data)
                # print("point%")
                break
            data_batches += new_data

        # 记录时间 p5
        point_5 = time.time()
        time_3 = point_5 - point_4
        print("downloading time: {}".format(time_3))

        print("processing...")
        # load
        data_text = json.loads(data_batches)
        cut_point = json.loads(cut_point)
        predictions = np.array(data_text)
        cut_point = int(cut_point)
        predictions = inference(sess, predictions, cut_point)

        # 记录时间 p6
        point_6 = time.time()
        time_4 = point_6 - point_5
        print("processing time: {}".format(time_4))

        print('Postprocessinasg...')
        # out_put images
        output_image = postprocessing(predictions, input_img_path, score_threshold, iou_threshold, input_height,
                                      input_width)
        # 记录时间 p7
        point_7 = time.time()
        time_5 = point_7 - point_6
        time_backend = time_3 + time_4 + time_5

        cv2.imwrite(output_image_path, output_image)
