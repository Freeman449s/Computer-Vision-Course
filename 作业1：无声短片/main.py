import cv2
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont

FRAME_SIZE = (480, 640, 3)
WINDOW_NAME = "Animation"
FPS = 25
FRAME_DURATION = int(1000 / FPS)


def initialize() -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)


def opening() -> None:
    LOGO_COLOR = (0, 171, 255)
    INFO_COLOR = (200, 200, 200)
    fontScale = 1
    bottom_left_for_logo = (0, 0)
    INFO_POSITION = (110, 360)
    # INFO_FONT_SCALE = 1
    THICKNESS = 3
    # logo缩小，同时画面渐暗
    nSeconds = 3
    for i in range(1, nSeconds * FPS + 1):
        frame = np.zeros(FRAME_SIZE, np.uint8)
        fontScale = math.log(800 / i)
        bottom_left_for_logo = (int(320 - 10 * fontScale), 240)
        colorScale = math.log(nSeconds * FPS / i, nSeconds * FPS)
        logoColor = (int(LOGO_COLOR[0] * colorScale), int(LOGO_COLOR[1] * colorScale), int(LOGO_COLOR[2] * colorScale))
        # THICKNESS = int(math.log(1600 / (i + 1)))
        cv2.putText(frame, "T", bottom_left_for_logo, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale, logoColor,
                    THICKNESS)  # 参数：帧，文本，左下角坐标，字体，缩放，颜色，线条粗细
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(FRAME_DURATION)
    # 展示校徽
    schoolBadge = cv2.imread("images\\School Badge3.jpg")
    frame = schoolBadge
    nSeconds = 1
    for i in range(1, nSeconds * FPS + 1):
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(FRAME_DURATION)
    black = np.zeros(FRAME_SIZE, np.uint8)
    nSeconds = 2
    for i in range(1, nSeconds * FPS + 1):
        weight = math.log(nSeconds * FPS / i, nSeconds * FPS)
        frame = cv2.addWeighted(schoolBadge, weight, black, 1 - weight, 0)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(FRAME_DURATION)
    # 展示头像与信息，同时画面渐暗
    headImage = cv2.imread("images\\Head Image.jpg")
    # 利用PIL显示中文
    cv2Frame = cv2.cvtColor(headImage, cv2.COLOR_BGR2RGB)  # 通道顺序转换
    pilFrame = Image.fromarray(cv2Frame)  # 转为PIL图像
    draw = ImageDraw.Draw(pilFrame)  # 创建一个绘图对象
    fontStyle = ImageFont.truetype("font\\Dengb.TTF", size=48, encoding="UTF-8")  # 创建字体对象，字体使用华文新魏
    draw.text(INFO_POSITION, "3180101042 唐敏哲", INFO_COLOR, font=fontStyle)
    infoImg = cv2.cvtColor(np.asarray(pilFrame), cv2.COLOR_RGB2BGR)  # 转换回cv图像
    nSeconds = 1
    for i in range(1, nSeconds * FPS + 1):
        cv2.imshow(WINDOW_NAME, infoImg)
        cv2.waitKey(FRAME_DURATION)
    nSeconds = 2
    for i in range(1, nSeconds * FPS + 1):
        weight = math.log(nSeconds * FPS / i, nSeconds * FPS)
        frame = cv2.addWeighted(infoImg, weight, black, 1 - weight, 0)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(FRAME_DURATION)
    cv2.waitKey(0)


def drawCircle_progressive(frame: np.array, center: tuple, radius: float, nSteps: int, color: tuple,
                           thickness: int) -> None:
    for i in range(0, nSteps):
        startRad = i / nSteps * (2 * math.pi)
        endRad = (i + 1) / nSteps * (2 * math.pi)
        startPos = (int(center[0] + radius * math.sin(startRad)), int(center[1] - radius * math.cos(startRad)))
        endPos = (int(center[0] + radius * math.sin(endRad)), int(center[1] - radius * math.cos(endRad)))
        cv2.line(frame, startPos, endPos, color, thickness)  # 注意：cv2的坐标是以(x,y)表示的
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(FRAME_DURATION)





def drawLine_progressive(frame: np.array, startPos: tuple, endPos: tuple, nSeconds: int, color: tuple,
                         thickness: int) -> None:
    TOTAL_STEPS = nSeconds * FPS
    for i in range(1, TOTAL_STEPS + 1):
        x = int(startPos[0] + i / TOTAL_STEPS * (endPos[0] - startPos[0]))
        y = int(startPos[1] + i / TOTAL_STEPS * (endPos[1] - startPos[1]))
        cv2.line(frame, startPos, (x, y), color, thickness)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(FRAME_DURATION)


BLACK = np.zeros(FRAME_SIZE, np.uint8)
drawLine_progressive(BLACK, (0, 240), (320, 240), 3, (255, 255, 255), 1)
cv2.waitKey(0)
