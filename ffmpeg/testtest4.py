from openvino.inference_engine import IENetwork, IEPlugin
import subprocess as sp
import cv2
import time
import numpy as np

model_xml = "/home/adv/szk/person-detection-0100.xml.bk"
model_bin = "/home/adv/szk/person-detection-0100.bin.bk"
#model_dir = "/home/adv/intel/openvino_2020.4.287/deployment_tools/model_optimizer/person/ssd_mobile_16FP/"
#model_xml = model_dir + "frozen_inference_graph.xml"
#model_bin = model_dir + "frozen_inference_graph.bin"

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

rtmpUrl="rtmp://192.168.0.118:1935/live/1"
camera_path = "rtsp://admin:jishu123@192.168.0.115:554/h264/ch1/main/av_stream"
cap = cv2.VideoCapture(0)

# Get video information
cap.set(cv2.CAP_PROP_FPS,25)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# ffmpeg command
command = ['ffmpeg',
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', "{}x{}".format(width, height),
           '-r', str(fps),
           '-i', '-',
           '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv',
           rtmpUrl]

# 管道配置
p = sp.Popen(command, stdin=sp.PIPE)

plugin = IEPlugin(device="MYRIAD")
net = IENetwork(model=model_xml, weights=model_bin)
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
n, c, h, w = net.inputs[input_blob].shape
print(n,c,h,w)
exec_net = plugin.load(network=net, num_requests=1)


imgDetect = ImgDetect()

# read webcamera
while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        print("Opening camera is failed")
        break


    oh,ow,_ = frame.shape
    image = cv2.resize(frame, (w, h))
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    print(image.shape)

    #oh,ow,_ = frame.shape
    #image = cv2.resize(frame, (w, h))
    #print(type(image))
    #image = np.expand_dims(image, axis=0)
    #image = image.transpose(image,(0, 3, 1, 2))
    




    #res = exec_net.infer(inputs={input_blob: image}).get("DetectionOutput")
    res = exec_net.infer(inputs={input_blob: image})
    print(res)
    draw_img = imgDetect.draw_boxes(frame, ow, oh, res, fontScale=1)
    #time.sleep(0.032)
    p.stdin.write(draw_img.tostring())
