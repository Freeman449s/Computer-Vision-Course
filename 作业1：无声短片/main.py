import cv2
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont

FRAME_SIZE = (480, 640, 3)
WINDOW_NAME = "Animation"
FPS = 25
FRAME_DURATION = int(1000 / FPS)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DASH_LINE_LENGTH = 96  # 允许跨越同向分隔线的长度
DASH_LINE_WIDTH = 16
LINE_INTERVAL = 32  # 允许跨越同向分隔线之间的距离
LINE_MOVE_SPEED = 360  # 分隔线每秒移动的距离（像素）
UPPER_CAR_INIT_POS = (160, 120)
LOWER_CAR_INIT_POS = (240, 360)
UPPER_CAR_END_POS = (240, 120)
LOWER_CAR_END_POS = (160, 360)


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
    nSeconds = 2
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


def drawLine_progressive(frame: np.array, startPos: tuple, endPos: tuple, nSeconds: float, color: tuple,
                         thickness: int) -> None:
    TOTAL_STEPS = round(nSeconds * FPS)
    for i in range(1, TOTAL_STEPS + 1):
        x = int(startPos[0] + i / TOTAL_STEPS * (endPos[0] - startPos[0]))
        y = int(startPos[1] + i / TOTAL_STEPS * (endPos[1] - startPos[1]))
        cv2.line(frame, startPos, (x, y), color, thickness)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(FRAME_DURATION)


def drawRect_progressive(frame: np.array, center: tuple, xSpan: int, ySpan: int, xNSeconds: float, yNSeconds: float,
                         color: tuple, thickness: int) -> None:
    LEFT_TOP = (round(center[0] - xSpan / 2), round(center[1] - ySpan / 2))
    RIGHT_TOP = (round(center[0] + xSpan / 2), round(center[1] - ySpan / 2))
    LEFT_BOTTOM = (round(center[0] - xSpan / 2), round(center[1] + ySpan / 2))
    RIGHT_BOTTOM = (round(center[0] + xSpan / 2), round(center[1] + ySpan / 2))
    drawLine_progressive(frame, LEFT_TOP, RIGHT_TOP, xNSeconds, color, thickness)
    drawLine_progressive(frame, RIGHT_TOP, RIGHT_BOTTOM, yNSeconds, color, thickness)
    drawLine_progressive(frame, RIGHT_BOTTOM, LEFT_BOTTOM, xNSeconds, color, thickness)
    drawLine_progressive(frame, LEFT_BOTTOM, LEFT_TOP, yNSeconds, color, thickness)


def drawCar_progressive(frame: np.array, center: tuple, scale: float, nSeconds: float, color: tuple,
                        thickness: int) -> None:
    CAR_WIDTH = round(108 * scale)
    CAR_HEIGHT = round(64 * scale)
    RADIUS = 8 * scale
    LEFT_TOP = (round(center[0] - CAR_WIDTH / 2), round(center[1] - CAR_HEIGHT / 2))
    LEFT_BOTTOM = (round(center[0] - CAR_WIDTH / 2), round(center[1] + CAR_HEIGHT / 2))
    RIGHT_TOP = (round(center[0] + CAR_WIDTH / 2), round(center[1] - CAR_HEIGHT / 2))
    RIGHT_BOTTOM = (round(center[0] + CAR_WIDTH / 2), round(center[1] + CAR_HEIGHT / 2))
    VERTICAL_DURATION = CIRCLE_DURATION = nSeconds / 8
    CIRCLE_NSTEPS = round(CIRCLE_DURATION * FPS)
    HORIZONTAL_DURATION = nSeconds / 4
    drawLine_progressive(frame, LEFT_TOP, LEFT_BOTTOM, VERTICAL_DURATION, color, thickness)
    drawLine_progressive(frame, LEFT_TOP, RIGHT_TOP, HORIZONTAL_DURATION, color, thickness)
    drawLine_progressive(frame, RIGHT_TOP, RIGHT_BOTTOM, VERTICAL_DURATION, color, thickness)
    drawLine_progressive(frame, LEFT_BOTTOM, (round(LEFT_BOTTOM[0] + CAR_WIDTH / 6), LEFT_BOTTOM[1]),
                         HORIZONTAL_DURATION / 4, color, thickness)
    drawCircle_progressive(frame, (round(LEFT_BOTTOM[0] + CAR_WIDTH / 6 + RADIUS), LEFT_BOTTOM[1]), RADIUS,
                           CIRCLE_NSTEPS, color, thickness)
    drawLine_progressive(frame, (round(LEFT_BOTTOM[0] + CAR_WIDTH / 6 + RADIUS * 2), LEFT_BOTTOM[1]),
                         (round(RIGHT_BOTTOM[0] - CAR_WIDTH / 6 - RADIUS * 2), RIGHT_BOTTOM[1]),
                         HORIZONTAL_DURATION / 2, color, thickness)
    drawCircle_progressive(frame, (round(RIGHT_BOTTOM[0] - CAR_WIDTH / 6 - RADIUS), RIGHT_BOTTOM[1]), RADIUS,
                           CIRCLE_NSTEPS, color, thickness)
    drawLine_progressive(frame, (round(RIGHT_BOTTOM[0] - CAR_WIDTH / 6), RIGHT_BOTTOM[1]), RIGHT_BOTTOM,
                         HORIZONTAL_DURATION / 4, color, thickness)


def drawCar(frame: np.array, center: tuple, scale: float, color: tuple, thickness: int) -> None:
    """
    立即绘制小车。
    注意：函数只更新帧，不会将帧显示出来，必须手动调用imshow()和wiatKey()\n
    :param frame: 待修改的帧
    :param center: 小车的中心
    :param scale: 小车的缩放比
    :param color: 小车的颜色
    :param thickness: 笔画的粗细
    :return: 无返回值
    """
    CAR_WIDTH = round(108 * scale)
    CAR_HEIGHT = round(64 * scale)
    RADIUS = 8 * scale
    LEFT_TOP = (round(center[0] - CAR_WIDTH / 2), round(center[1] - CAR_HEIGHT / 2))
    LEFT_BOTTOM = (round(center[0] - CAR_WIDTH / 2), round(center[1] + CAR_HEIGHT / 2))
    RIGHT_TOP = (round(center[0] + CAR_WIDTH / 2), round(center[1] - CAR_HEIGHT / 2))
    RIGHT_BOTTOM = (round(center[0] + CAR_WIDTH / 2), round(center[1] + CAR_HEIGHT / 2))
    cv2.line(frame, LEFT_TOP, RIGHT_TOP, color, thickness)
    cv2.line(frame, LEFT_TOP, LEFT_BOTTOM, color, thickness)
    cv2.line(frame, RIGHT_TOP, RIGHT_BOTTOM, color, thickness)
    cv2.line(frame, LEFT_BOTTOM, (round(LEFT_BOTTOM[0] + CAR_WIDTH / 6), LEFT_BOTTOM[1]), color, thickness)
    cv2.line(frame, (round(LEFT_BOTTOM[0] + CAR_WIDTH / 6 + RADIUS * 2), LEFT_BOTTOM[1]),
             (round(RIGHT_BOTTOM[0] - CAR_WIDTH / 6 - RADIUS * 2), RIGHT_BOTTOM[1]), color, thickness)
    cv2.line(frame, (round(RIGHT_BOTTOM[0] - CAR_WIDTH / 6), RIGHT_BOTTOM[1]), RIGHT_BOTTOM, color, thickness)
    cv2.circle(frame, (round(LEFT_BOTTOM[0] + CAR_WIDTH / 6 + RADIUS), LEFT_BOTTOM[1]), RADIUS, color, thickness)
    cv2.circle(frame, (round(RIGHT_BOTTOM[0] - CAR_WIDTH / 6 - RADIUS), RIGHT_BOTTOM[1]), RADIUS, color, thickness)


def drawDashLines(frame: np.array, firstLineCenterX: int) -> int:
    if (firstLineCenterX <= -DASH_LINE_LENGTH / 2):
        firstLineCenterX += DASH_LINE_LENGTH + LINE_INTERVAL
    x = firstLineCenterX
    while x <= 640 + DASH_LINE_LENGTH / 2:
        # drawRect_progressive(frame, (x, 240), DASH_LINE_LENGTH, DASH_LINE_WIDTH, 1, 0.5, WHITE, 2)x
        cv2.rectangle(frame, (x - round(DASH_LINE_LENGTH / 2), 240 - round(DASH_LINE_WIDTH / 2)),
                      (round(x + DASH_LINE_LENGTH / 2), 240 + round(DASH_LINE_WIDTH / 2)), WHITE, 2)
        x += DASH_LINE_LENGTH + LINE_INTERVAL
    return firstLineCenterX - round(LINE_MOVE_SPEED / FPS)


def setupScene() -> None:
    """
    布置场景\n
    :return: 无返回值
    """
    frame = np.zeros(FRAME_SIZE, np.uint8)
    x = 0
    while x <= 640 + DASH_LINE_LENGTH / 2:
        drawRect_progressive(frame, (x, 240), DASH_LINE_LENGTH, DASH_LINE_WIDTH, 1, 0.5, WHITE, 2)
        x += DASH_LINE_LENGTH + LINE_INTERVAL
    drawCar_progressive(frame, UPPER_CAR_INIT_POS, 1, 4, WHITE, 2)
    drawCar_progressive(frame, LOWER_CAR_INIT_POS, 1, 4, WHITE, 2)


def mainContent() -> None:
    # 第1阶段，车保持相对静止
    nSeconds = 2
    firstLineCenterX = 0
    for i in range(1, nSeconds * FPS + 1):
        frame = np.zeros(FRAME_SIZE, np.uint8)
        drawCar(frame, UPPER_CAR_INIT_POS, 1, WHITE, 2)
        drawCar(frame, LOWER_CAR_INIT_POS, 1, WHITE, 2)
        firstLineCenterX = drawDashLines(frame, firstLineCenterX)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(40)
    # 第2阶段，上方车超越
    nSeconds = 1
    for i in range(1, nSeconds * FPS + 1):
        frame = np.zeros(FRAME_SIZE, np.uint8)
        upperCarCenterX = round(
            i / (nSeconds * FPS) * (UPPER_CAR_END_POS[0] - UPPER_CAR_INIT_POS[0]) + UPPER_CAR_INIT_POS[0])
        lowerCarCenterX = round(
            i / (nSeconds * FPS) * (LOWER_CAR_END_POS[0] - LOWER_CAR_INIT_POS[0]) + LOWER_CAR_INIT_POS[0])
        drawCar(frame, (upperCarCenterX, UPPER_CAR_INIT_POS[1]), 1, WHITE, 2)
        drawCar(frame, (lowerCarCenterX, LOWER_CAR_INIT_POS[1]), 1, WHITE, 2)
        firstLineCenterX = drawDashLines(frame, firstLineCenterX)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(40)
    # 第3阶段，相对静止
    nSeconds = 1
    for i in range(1, nSeconds * FPS + 1):
        frame = np.zeros(FRAME_SIZE, np.uint8)
        drawCar(frame, UPPER_CAR_END_POS, 1, WHITE, 2)
        drawCar(frame, LOWER_CAR_END_POS, 1, WHITE, 2)
        firstLineCenterX = drawDashLines(frame, firstLineCenterX)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(40)
    # 第4阶段，下方车超越
    nSeconds = 1
    for i in range(1, nSeconds * FPS + 1):
        frame = np.zeros(FRAME_SIZE, np.uint8)
        upperCarCenterX = round(
            i / (nSeconds * FPS) * (UPPER_CAR_INIT_POS[0] - UPPER_CAR_END_POS[0]) + UPPER_CAR_END_POS[0])
        lowerCarCenterX = round(
            i / (nSeconds * FPS) * (LOWER_CAR_INIT_POS[0] - LOWER_CAR_END_POS[0]) + LOWER_CAR_END_POS[0])
        drawCar(frame, (upperCarCenterX, UPPER_CAR_INIT_POS[1]), 1, WHITE, 2)
        drawCar(frame, (lowerCarCenterX, LOWER_CAR_INIT_POS[1]), 1, WHITE, 2)
        firstLineCenterX = drawDashLines(frame, firstLineCenterX)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(40)
    # 第5阶段，相对静止
    nSeconds = 1
    for i in range(1, nSeconds * FPS + 1):
        frame = np.zeros(FRAME_SIZE, np.uint8)
        drawCar(frame, UPPER_CAR_INIT_POS, 1, WHITE, 2)
        drawCar(frame, LOWER_CAR_INIT_POS, 1, WHITE, 2)
        firstLineCenterX = drawDashLines(frame, firstLineCenterX)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(40)
    # 第6阶段，上方车超越
    nSeconds = 1
    for i in range(1, nSeconds * FPS + 1):
        frame = np.zeros(FRAME_SIZE, np.uint8)
        upperCarCenterX = round(
            i / (nSeconds * FPS) * (UPPER_CAR_END_POS[0] - UPPER_CAR_INIT_POS[0]) + UPPER_CAR_INIT_POS[0])
        lowerCarCenterX = round(
            i / (nSeconds * FPS) * (LOWER_CAR_END_POS[0] - LOWER_CAR_INIT_POS[0]) + LOWER_CAR_INIT_POS[0])
        drawCar(frame, (upperCarCenterX, UPPER_CAR_INIT_POS[1]), 1, WHITE, 2)
        drawCar(frame, (lowerCarCenterX, LOWER_CAR_INIT_POS[1]), 1, WHITE, 2)
        firstLineCenterX = drawDashLines(frame, firstLineCenterX)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(40)
    # 第7阶段，另一辆车进入视野，画面渐亮
    nSeconds = 0.2
    collisionCarCenterX = 640 + 32
    for i in range(1, round(nSeconds * FPS + 1)):
        frame = np.zeros(FRAME_SIZE, np.uint8)
        drawCar(frame, (collisionCarCenterX, UPPER_CAR_INIT_POS[1]), 1, WHITE, 2)
        collisionCarCenterX -= round(LINE_MOVE_SPEED / FPS)
        drawCar(frame, UPPER_CAR_END_POS, 1, WHITE, 2)
        drawCar(frame, LOWER_CAR_END_POS, 1, WHITE, 2)
        firstLineCenterX = drawDashLines(frame, firstLineCenterX)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(40)
    nSeconds = 0.5
    totalSteps = int(nSeconds * FPS)
    WHITE_FRAME = np.full(FRAME_SIZE, 255, np.uint8)
    for i in range(1, totalSteps + 1):
        weight = 1 - i / totalSteps
        frame = np.zeros(FRAME_SIZE, np.uint8)
        frame = cv2.addWeighted(frame, weight, WHITE_FRAME, 1 - weight, 0)
        drawCar(frame, (collisionCarCenterX, UPPER_CAR_INIT_POS[1]), 1, WHITE, 2)
        collisionCarCenterX -= round(LINE_MOVE_SPEED / FPS)
        cv2.putText(frame, "!", (UPPER_CAR_END_POS[0] + 72, UPPER_CAR_END_POS[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 2,
                    WHITE, 2)
        drawCar(frame, UPPER_CAR_END_POS, 1, WHITE, 2)
        drawCar(frame, LOWER_CAR_END_POS, 1, WHITE, 2)
        firstLineCenterX = drawDashLines(frame, firstLineCenterX)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(40)


def ending() -> None:
    nSeconds = 1
    totalFrames = round(nSeconds * FPS)
    for i in range(1, totalFrames + 1):
        frame = np.full(FRAME_SIZE, 255, np.uint8)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(40)
    nSeconds = 0.5
    MAX_FONT_SIZE = 48
    fps = 50
    frameDuration = round(1000 / fps)
    totalFrames = round(nSeconds * fps)
    for i in range(1, totalFrames + 1):
        frame = np.full(FRAME_SIZE, 255, np.uint8)
        cv2Frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 通道顺序转换
        pilFrame = Image.fromarray(cv2Frame)  # 转为PIL图像
        draw = ImageDraw.Draw(pilFrame)  # 创建一个绘图对象
        fontSize = round(MAX_FONT_SIZE * i / totalFrames)
        fontStyle = ImageFont.truetype("font\\STXINWEI.TTF", size=fontSize, encoding="UTF-8")  # 创建字体对象，字体使用华文新魏
        x = round(320 - i / totalFrames * MAX_FONT_SIZE * 2.5)
        y = round(240 - i / totalFrames * MAX_FONT_SIZE * 1.5)
        draw.text((x, y), "安全无小事\n\n莫开英雄车", BLACK, font=fontStyle)
        frame = cv2.cvtColor(np.asarray(pilFrame), cv2.COLOR_RGB2BGR)  # 转换回cv图像
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(frameDuration)


opening()
setupScene()
mainContent()
ending()
cv2.waitKey(0)
