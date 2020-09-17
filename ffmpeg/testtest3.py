import subprocess as sp
import cv2

#rtmpUrl="rtmp://10.180.149.65:1935/live/1"
rtmpUrl="rtmp://113680.livepush.myqcloud.com/live/b1?txSecret=feef165f4e767d7933398b76d8063922&txTime=5F747B4B"
camera_path = "rtsp://admin:jishu123@192.168.0.115:554/h264/ch1/main/av_stream"
cap = cv2.VideoCapture(0)

# Get video information
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

# read webcamera
while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        print("Opening camera is failed")
        break

    p.stdin.write(frame.tostring())
