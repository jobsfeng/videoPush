#! /usr/bin/env python
# coding=utf-8
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin # 导入openvino库
input_size = 256
# 搭建网络，调用IENetwork，返回net对象，参数为xml和bin文件的路径
model_xml_CPU = r'person-detection-0100.xml.bk'
model_bin_CPU = r'person-detection-0100.bin.bk'
net = IENetwork(model=model_xml_CPU, weights=model_bin_CPU)
# 定义输入输出
input_blob = next(iter(net.inputs)) # 迭代器
out_blob   = next(iter(net.outputs))
print(input_blob, out_blob)
n, c, h, w = net.inputs[input_blob].shape
print(n,c,h,w)
# 加载设备，调用IEPlugin，返回设备对象，参数为设备名，如CPU、GPU、MYRIAD
plugin = IEPlugin(device='MYRIAD')
# 加载网络，调用设备对象的load方法，返回执行器，参数为网络
exec_net = plugin.load(network=net) 
print('load ok!')

#图片处理方式
img_path = "person_img.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(input_size,input_size))
img = np.array(img)
img = np.expand_dims(img,axis=0)
img = np.transpose(img,(0,3,1,2))
import time
for i in range(100): 
# 推理模型，调用执行器的infer方法，返回模型输出，输入为格式化图像
    start = time.time()
    outputs = exec_net.infer(inputs={input_blob:img}) 
    print(time.time()-start)
print(outputs['detection_out'][0,0,0,:])
