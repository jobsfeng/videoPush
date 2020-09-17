from openvino.inference_engine import IENetwork, IEPlugin
import cv2
import queue
import os
import numpy as np
from threading import Thread
import datetime, _thread
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






def run():

    plugin = IEPlugin(device="MYRIAD")
    #plugin = IEPlugin(device="CPU")
    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    n, c, h, w = net.inputs[input_blob].shape
    exec_net = plugin.load(network=net, num_requests=1)

    del net

    imgDetect = ImgDetect()

    vid = cv2.VideoCapture(0)
    # Get video information
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    print("input fps is :                                            *****************************" + str(fps))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    else:
        print("6666")
    while (vid.isOpened()):
        return_value, frame = vid.read()
        #inf_start = time.time()
        oh, ow, _ = frame.shape
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        inf_start = time.time()
        res = exec_net.infer(inputs={input_blob: image}).get("DetectionOutput")
        draw_img = imgDetect.draw_boxes(frame, ow, oh, res, fontScale=1)
        inf_end = time.time()
        det_time = inf_end - inf_start
        print("Inference time: {:.3f} ms, FPS:{:.3f}".format(det_time * 1000, 1000 / (det_time*1000 + 1)))

    del exec_net
    del plugin


if __name__ == '__main__':
    run()









