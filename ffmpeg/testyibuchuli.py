import sys
import cv2


from openvino.inference_engine import IENetwork, IEPlugin
import subprocess as sp
rtmpUrl = "rtmp://192.168.0.118:1935/live/1"
camera_path = "rtsp://admin:jishu123@192.168.0.115:554/h264/ch1/main/av_stream"



model_dir = "/home/adv/intel/openvino_2020.4.287/deployment_tools/model_optimizer/person/ssd_mobile_16FP/"
model_xml = model_dir + "frozen_inference_graph.xml"
model_bin = model_dir + "frozen_inference_graph.bin"

def human_detect_demo():
    plugin = IEPlugin(device="MYRIAD")
    net = IENetwork(model=model_xml, weights=model_bin)

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    print("input_blob",type(input_blob))
    n, c, h, w = net.inputs[input_blob].shape
    exec_net = plugin.load(network=net, num_requests=2)

    del net


    cap= cv2.VideoCapture(camera_path)
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

    cur_request_id = 0
    next_request_id = 1

    is_async_mode = True

    ret, frame = cap.read()

    while cap.isOpened():
        if is_async_mode:
            ret, next_frame = cap.read()
        else:
            ret, frame = cap.read()
        if not ret:
            break
        initial_w = cap.get(3)
        initial_h = cap.get(4)

        # 开启同步或者异步执行模式
        if is_async_mode:
            in_frame = cv2.resize(next_frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW 
            in_frame = in_frame.reshape((n,c, h, w))
            exec_net.start_async(request_id=next_request_id, inputs={input_blob: in_frame})
        else:
            in_frame = cv2.resize(frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            # 获取网络输出
            res = exec_net.requests[cur_request_id].outputs[out_blob]
            # print(res.shape)
            # print(res)
            tp = ['person']
            res = res[0][0]
            for r in res:
                if r[2] <= 0.3:
                    continue
                if r[1] == 1:
                    color = [255, 0, 0]
                    # name = tp[int(r[1]-1)]
                    name = tp[0]
                    x1 = int(r[3] * h)
                    y1 = int(r[4] * w)
                    x2 = int(r[5] * h)
                    y2 = int(r[6] * w)
                    txt = '{}{:.1f}'.format(name, r[2])
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, txt, (x1, y1 - 5), font, 1, (255, 255, 255), thickness=1,
                                lineType=cv2.LINE_AA)
        p.stdin.write(frame.tostring())



        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            frame = next_frame

        key = cv2.waitKey(1)
        if key == 27:
            break

    del exec_net
    del plugin



if __name__ == '__main__':
    sys.exit(human_detect_demo() or 0)
