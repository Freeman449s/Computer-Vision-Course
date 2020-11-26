"""
闲置的函数
"""
import numpy as np
import cv2

FRAME_SIZE = (480, 640, 3)

# 将每个像素的像素值乘以scale，再转为np.uint8类型
# 用于制作渐暗效果
# 因性能较差而闲置
def pixelValueScale(frame: np.array, scale: float) -> np.array:
    ret = np.zeros(frame.shape, np.uint8)
    for i in range(0, frame.shape[0]):
        for j in range(0, frame.shape[1]):
            for k in range(0, frame.shape[2]):
                ret[i][j][k] = np.uint8(scale * frame[i][j][k])
    return ret


def camCapture():
    cam = cv2.VideoCapture(0)  # 监听0号摄像头
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取摄像头返回帧的宽度和高度
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # fourcc: Four-Character Codec, 四位编码方式；mp4v是mp4的编码方式
    out = cv2.VideoWriter("Cam Capture.mp4", fourcc, 25.0, (width, height))  # 创建输出对象
    cam.open(0)  # 开始监听
    while True:
        successful, frame = cam.read()  # 返回摄像头捕获的下一帧，以及成功与否
        if successful:
            cv2.imshow("Camera Capture", frame)
            out.write(frame)
        keyID = cv2.waitKey(40)  # 每40ms刷新一次，25fps
        if keyID == 27:  # 按下Esc则退出
            break
    cam.release()  # 释放摄像机资源
    out.release()  # 释放文件资源
    cv2.destroyAllWindows()  # 销毁所有窗口


def animation():
    frame = np.zeros(FRAME_SIZE, np.uint8)
    cv2.line(frame, (0, 0), (640, 480), (0, 255, 0), 3)  # 参数：图像，起点（左上角为(0,0)），终点，颜色（BGR），宽度
    cv2.imshow("Animation", frame)
    cv2.waitKey(0)
