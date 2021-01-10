"""
闲置的函数
"""
import numpy as np
import cv2
import math

FRAME_SIZE = (480, 640, 3)
WINDOW_NAME = "Animation"
FPS = 25
FRAME_DURATION = int(1000 / FPS)


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


def drawRectangle_progressive(frame: np.array, center: tuple, xSpan: int, ySpan: int, color: tuple,
                              thickness: int) -> None:
    """
    渐进绘制矩形的函数，因存在大量重复代码而弃用\n
    :param frame:
    :param center:
    :param xSpan:
    :param ySpan:
    :param color:
    :param thickness:
    :return:
    """
    if xSpan > ySpan:
        xDuration = 1
        yDuration = 0.5
    else:
        xDuration = 0.5
        yDuration = 1
    TOP_LEFT = (int(center[0] - xSpan / 2), int(center[1] - ySpan / 2))
    TOP_RIGHT = (int(center[0] + xSpan / 2), int(center[1] - ySpan / 2))
    BOTTOM_RIGHT = (int(center[0] + xSpan / 2), int(center[1] + ySpan / 2))
    BOTTOM_LEFT = (int(center[0] - xSpan / 2), int(center[1] + ySpan / 2))
    startPos = TOP_LEFT
    for i in range(1, math.floor(xDuration * FPS + 1)):
        endPos = (int(startPos[0] + xSpan * i / (xDuration * FPS)), startPos[1])
        cv2.line(frame, startPos, endPos, color, thickness)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(FRAME_DURATION)
    startPos = TOP_RIGHT
    for j in range(1, math.floor(yDuration * FPS + 1)):
        endPos = (startPos[0], int(startPos[1] + ySpan * j / (yDuration * FPS)))
        cv2.line(frame, startPos, endPos, color, thickness)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(FRAME_DURATION)
    startPos = BOTTOM_RIGHT
    for i in range(1, math.floor(xDuration * FPS + 1)):
        endPos = (int(startPos[0] - xSpan * i / (xDuration * FPS)), startPos[1])
        cv2.line(frame, startPos, endPos, color, thickness)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(FRAME_DURATION)
    startPos = BOTTOM_LEFT
    for j in range(1, math.floor(yDuration * FPS + 1)):
        endPos = (startPos[0], int(startPos[1] - ySpan * j / (yDuration * FPS)))
        cv2.line(frame, startPos, endPos, color, thickness)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(FRAME_DURATION)
