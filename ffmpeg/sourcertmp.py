from openvino.inference_engine import IENetwork, IEPlugin
import cv2
import queue
import os
import numpy as np
from threading import Thread
import datetime,_thread
import subprocess as sp
import time

model_dir = "/home/adv/intel/openvino_2020.4.287/deployment_tools/model_optimizer/person/ssd_mobile_16FP/"
model_xml = model_dir + "frozen_inference_graph.xml"
model_bin = model_dir + "frozen_inference_graph.bin"

class ImgDetect:

    def draw_boxes(self, img_cv, h, w, res, score=0.3, fontScale=1):

        # tp = ['hat','person']
        tp = ['person']
        res = res[0][0]
        for r in res:
            if r[2] <= score:
                continue
            if r[1] == 1:
                c = [255, 0, 0]
                # name = tp[int(r[1]-1)]
                name = tp[0]
                x1 = int(r[3] * h)
                y1 = int(r[4] * w)
                x2 = int(r[5] * h)
                y2 = int(r[6] * w)
                txt = '{}{:.1f}'.format(name, r[2])
                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.rectangle(img_cv, (x1, y1), (x2, y2), c, 2)
                cv2.putText(img_cv, txt, (x1, y1 - 5), font, fontScale, (255, 255, 255), thickness=1,
                            lineType=cv2.LINE_AA)
        return img_cv

# 使用线程锁，防止线程死锁
mutex = _thread.allocate_lock()
# 存图片的队列
frame_queue = queue.Queue()
# 推流的地址，前端通过这个地址拉流，主机的IP，2019是ffmpeg在nginx中设置的端口号
rtmpUrl="rtmp://192.168.0.118:1935/live/1"
# 用于推流的配置,参数比较多，可网上查询理解
command=[]
 
def Video(vid,imgDetect,n, c, h, w,exec_net,input_blob,out_blob):
# 调用相机拍图的函数
    #vid = cv2.VideoCapture("rtsp://admin:jishu123@192.168.0.115:554/h264/ch1/main/av_stream")
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    else:
        print("6666")
    while (vid.isOpened()):
        return_value, frame = vid.read()
        #oh,ow,_ = frame.shape
        #image = cv2.resize(frame, (w, h))
        #image = image.transpose((2, 0, 1))
        #image = image.reshape((n, c, h, w))
        #res = exec_net.infer(inputs={input_blob: image}).get("DetectionOutput")
        #draw_img = imgDetect.draw_boxes(frame, ow, oh, res, fontScale=1)
        # 原始图片推入队列中
        frame_queue.put(frame)
 
 
def push_frame(imgDetect,n, c, h, w,exec_net,input_blob,out_blob):
    # 推流函数
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    #prev_time = time()
 
    # 防止多线程时 command 未被设置
    while True:
        if len(command) > 0:
            # 管道配置，其中用到管道
            p = sp.Popen(command, stdin=sp.PIPE)
            break
 
    
    while True:
        if frame_queue.empty() != True:
            #从队列中取出图片
            frame = frame_queue.get()
            #oh,ow,_ = frame.shape
            #image = cv2.resize(frame, (w, h))
            #image = image.transpose((2, 0, 1))
            #image = image.reshape((n, c, h, w))
            #res = exec_net.infer(inputs={input_blob: image}).get("DetectionOutput")
            #draw_img = imgDetect.draw_boxes(frame, ow, oh, res, fontScale=1)

          
            # write to pipe
            # 将处理后的图片通过管道推送到服务器上,image是处理后的图片
            p.stdin.write(frame.tostring())
 
def run():
     #使用两个线程处理
    
    plugin = IEPlugin(device="MYRIAD")
    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    n, c, h, w = net.inputs[input_blob].shape
    exec_net = plugin.load(network=net, num_requests=1)

    
    del net
    imgDetect = ImgDetect()

    vid = cv2.VideoCapture("rtsp://admin:jishu123@192.168.0.115:554/h264/ch1/main/av_stream")
    # Get video information
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    print("input fps is :                                            *****************************"+str(fps))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))    
    command.append('ffmpeg')
    command.append('-y')
    command.append('-f')
    command.append('rawvideo')
    command.append('-vcodec')
    command.append('rawvideo')
    command.append('-pix_fmt')
    command.append('bgr24')
    command.append('-s')
    command.append("{}x{}".format(width, height))
    command.append('-r')
    command.append(str(fps))
    command.append('-i')
    command.append('-')
    command.append('-c:v')
    command.append('libx264')
    command.append('-pix_fmt')
    command.append('yuv420p')
    command.append('-preset')
    command.append('ultrafast')
    command.append('-f')
    command.append('flv')
    command.append(rtmpUrl)

    
    thread1 = Thread(target=Video,args=(vid,imgDetect,n, c, h, w,exec_net,input_blob,out_blob))
    thread1.start()
    thread2 = Thread(target=push_frame,args=(imgDetect,n, c, h, w,exec_net,input_blob,out_blob))
    thread2.start()

    del exec_net
    del plugin
if __name__ == '__main__':
 
    run()
