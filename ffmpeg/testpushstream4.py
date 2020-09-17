import queue
import threading
import cv2 as cv
import subprocess as sp



frame_queue = queue.Queue()
command = []
# 自行设置
rtmpUrl = "rtmp://192.168.0.118:1935/live/1"
camera_path = "rtsp://admin:jishu123@192.168.0.115:554/h264/ch1/main/av_stream"
        
def read_frame():
    print("开启推流")
    cap = cv.VideoCapture(camera_path)
    # Get video information
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    #fps:12     width:1280   height:960     

    # ffmpeg command
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
    # read webcamera
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print("Opening camera is failed")
            # 说实话这里的break应该替换为：
            # cap = cv.VideoCapture(self.camera_path)
            # 因为我这俩天遇到的项目里出现断流的毛病
            # 特别是拉取rtmp流的时候！！！！
            print("stream is shutted down")
            break
        else:
            print("the video has been read to frame")
        # put frame into queue
        frame_queue.put(frame)

def push_frame():
    # 防止多线程时 command 未被设置
    while True:
        if len(command) > 0:
            # 管道配置
            p = sp.Popen(command, stdin=sp.PIPE)
            print("popen has been confed")
            break

    while True:
        if frame_queue.empty() != True:
            frame = frame_queue.get()
            # process frame
            # 你处理图片的代码
            # write to pipe
            p.stdin.write(frame.tostring())
                
def run():
    threads = [
        threading.Thread(target=read_frame,),
        threading.Thread(target=push_frame,)
    ]
    #[thread.setDaemon(True) for thread in threads]
    [thread.start() for thread in threads]
if __name__ == '__main__':
    run()

