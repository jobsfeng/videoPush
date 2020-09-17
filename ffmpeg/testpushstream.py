import queue
import threading
import cv2 as cv
import subprocess as sp



from openvino.inference_engine import IENetwork, IEPlugin

import os

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
                font = cv.FONT_HERSHEY_SIMPLEX

                cv.rectangle(img_cv, (x1, y1), (x2, y2), c, 2)
                cv.putText(img_cv, txt, (x1, y1 - 5), font, fontScale, (255, 255, 255), thickness=1,
                            lineType=cv.LINE_AA)
        return img_cv


class Live(object):
    def __init__(self):
        self.frame_queue = queue.Queue()
        self.command = ""
        # 自行设置
        self.rtmpUrl = "rtmp://adv:123456@192.168.0.118:1935/live"
        self.camera_path = "rtsp://admin:jishu123@192.168.0.115:554/h264/ch1/main/av_stream"
        
    def read_frame(self):
        print("开启推流")
        cap = cv.VideoCapture(self.camera_path)

        # Get video information
        fps = int(cap.get(cv.CAP_PROP_FPS))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # ffmpeg command
        self.command = ['ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', "{}x{}".format(width, height),
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-f', 'flv', 
                self.rtmpUrl]
        
        plugin = IEPlugin(device="CPU")
        net = IENetwork(model=model_xml, weights=model_bin)
        input_blob = next(iter(net.inputs))
        out_blob = next(iter(net.outputs))
        n, c, h, w = net.inputs[input_blob].shape
        exec_net = plugin.load(network=net, num_requests=1)

        imgDetect = ImgDetect()
        # read webcamera
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                print("Opening camera is failed")
                # 说实话这里的break应该替换为：
                # cap = cv.VideoCapture(self.camera_path)
                # 因为我这俩天遇到的项目里出现断流的毛病
                # 特别是拉取rtmp流的时候！！！！
                break

            oh,ow,_ = frame.shape
            image = cv.resize(frame, (w, h))
            image = image.transpose((2, 0, 1))
            image = image.reshape((n, c, h, w))
            res = exec_net.infer(inputs={input_blob: image}).get("DetectionOutput")
            draw_img = imgDetect.draw_boxes(frame, ow, oh, res, fontScale=1)

            # put frame into queue
            self.frame_queue.put(draw_img)

    def push_frame(self):
        # 防止多线程时 command 未被设置
        while True:
            if len(self.command) > 0:
                # 管道配置
                p = sp.Popen(self.command, stdin=sp.PIPE)
                break

        while True:
            if self.frame_queue.empty() != True:
                frame = self.frame_queue.get()
                # process frame
                # 你处理图片的代码
                # write to pipe
                p.stdin.write(frame.tostring())
                
    def run(self):
        threads = [
            threading.Thread(target=Live.read_frame, args=(self,)),
            threading.Thread(target=Live.push_frame, args=(self,))
        ]
        [thread.setDaemon(True) for thread in threads]
        [thread.start() for thread in threads]



if __name__ == '__main__':
    live=Live()
    live.run()
