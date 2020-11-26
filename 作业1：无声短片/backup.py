"""
一些函数的备份，避免对函数的更改造成异常
"""
import numpy as np
import cv2

def opening() -> None:
    LOGO_COLOR = (0, 171, 255)
    INFO_COLOR = (255, 255, 255)
    fontScale = 1
    bottom_left_for_logo = (0, 0)
    INFO_POSITION = (110, 360)
    # INFO_FONT_SCALE = 1
    THICKNESS = 3
    # logo缩小，同时画面渐暗
    nSeconds = 3
    for i in range(1, nSeconds * 25 + 1):
        frame = np.zeros(FRAME_SIZE, np.uint8)
        fontScale = math.log(800 / i)
        bottom_left_for_logo = (int(320 - 10 * fontScale), 240)
        colorScale = math.log(nSeconds * 25 / i, nSeconds * 25)
        logoColor = (int(LOGO_COLOR[0] * colorScale), int(LOGO_COLOR[1] * colorScale), int(LOGO_COLOR[2] * colorScale))
        # THICKNESS = int(math.log(1600 / (i + 1)))
        cv2.putText(frame, "T", bottom_left_for_logo, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale, logoColor,
                    THICKNESS)  # 参数：帧，文本，左下角坐标，字体，缩放，颜色，线条粗细
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(40)
    # 画面渐暗效果
    # nSeconds = 2
    # for i in range(1, nSeconds * 25 + 1):
    #     frame = np.zeros(FRAME_SIZE, np.uint8)
    #     colorScale = math.log(nSeconds * 25 / i, nSeconds * 25)
    #     logoColor = (int(LOGO_COLOR[0] * colorScale), int(LOGO_COLOR[1] * colorScale), int(LOGO_COLOR[2] * colorScale))
    #     cv2.putText(frame, "T", bottom_left_for_logo, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale, logoColor, THICKNESS)
    #     cv2.imshow(WINDOW_NAME, frame)
    #     cv2.waitKey(40)
    # 展示校徽
    schoolBadge = cv2.imread("images\\School Badge3.jpg")
    frame = schoolBadge
    nSeconds = 1
    for i in range(1, nSeconds * 25 + 1):
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(40)
    black = np.zeros(FRAME_SIZE, np.uint8)
    nSeconds = 2
    for i in range(1, nSeconds * 25 + 1):
        weight = math.log(nSeconds * 25 / i, nSeconds * 25)
        frame = cv2.addWeighted(schoolBadge, weight, black, 1 - weight, 0)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(40)
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
    for i in range(1, nSeconds * 25 + 1):
        cv2.imshow(WINDOW_NAME, infoImg)
        cv2.waitKey(40)
    nSeconds = 2
    for i in range(1, nSeconds * 25 + 1):
        weight = math.log(nSeconds * 25 / i, nSeconds * 25)
        frame = cv2.addWeighted(infoImg, weight, black, 1 - weight, 0)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(40)
    # for i in range(1, nSeconds * 25 + 1):
    #     frame = np.zeros(FRAME_SIZE, np.uint8)
    #     cv2.putText(frame, "T", bottom_left_for_logo, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale, LOGO_COLOR, THICKNESS)
    #     # 利用PIL显示中文
    #     cv2Frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 通道顺序转换
    #     pilFrame = Image.fromarray(cv2Frame)  # 转为PIL图像
    #     draw = ImageDraw.Draw(pilFrame)  # 创建一个绘图对象
    #     fontStyle = ImageFont.truetype("STXINWEI.TTF", size=48, encoding="UTF-8")  # 创建字体对象，字体使用华文新魏
    #     draw.text(INFO_POSITION, "3180101042 唐敏哲", INFO_COLOR, font=fontStyle)
    #     frame = cv2.cvtColor(np.asarray(pilFrame), cv2.COLOR_RGB2BGR)  # 转换回cv图像
    #     cv2.imshow(WINDOW_NAME, frame)
    #     cv2.waitKey(40)
    # # 应用渐暗效果
    # nSeconds = 2
    # for i in range(1, nSeconds * 25 + 1):
    #     colorScale = math.log(nSeconds * 25 / i, nSeconds * 25)
    #     logoColor = (int(LOGO_COLOR[0] * colorScale), int(LOGO_COLOR[1] * colorScale), int(LOGO_COLOR[2] * colorScale))
    #     infoColor = (int(INFO_COLOR[0] * colorScale), int(INFO_COLOR[1] * colorScale), int(INFO_COLOR[2] * colorScale))
    #     frame = np.zeros(FRAME_SIZE, np.uint8)
    #     cv2.putText(frame, "T", bottom_left_for_logo, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale, logoColor, THICKNESS)
    #     # 利用PIL显示中文
    #     cv2Frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 通道顺序转换
    #     pilFrame = Image.fromarray(cv2Frame)  # 转为PIL图像
    #     draw = ImageDraw.Draw(pilFrame)  # 创建一个绘图对象
    #     fontStyle = ImageFont.truetype("font\\STXINWEI.TTF", size=48, encoding="UTF-8")  # 创建字体对象，字体使用华文新魏
    #     draw.text(INFO_POSITION, "3180101042 唐敏哲", infoColor, font=fontStyle)
    #     frame = cv2.cvtColor(np.asarray(pilFrame), cv2.COLOR_RGB2BGR)  # 转换回cv图像
    #     # cv2.putText(frame, "3180101042 唐敏哲", BOTTOM_LEFT_FOR_INFO, cv2.FONT_HERSHEY_SIMPLEX, INFO_FONT_SCALE, LOGO_COLOR,
    #     #             THICKNESS)
    #     cv2.imshow(WINDOW_NAME, frame)
    #     cv2.waitKey(40)
    cv2.waitKey(0)