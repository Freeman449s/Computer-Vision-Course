import numpy as np
import cv2
import math
from PIL import Image
import Util
from enum import Enum

IMG_PATH = "Seal Test.jpg"
WINDOW_NAME = "HW2"
CANNY_T = (400, 480)
LINE_COLOR = (255, 0, 0)
CENTER_COLOR = (255, 0, 0)


class SuppressionMode(Enum):
    FOUR_CONN = 1
    EIGHT_CONN = 2


def lineDetection(img: np.ndarray, edges: np.ndarray) -> None:
    accumMat = np.zeros(((edges.shape[0] + edges.shape[1]) * 2 + 1, 180), int)
    for y in range(0, edges.shape[0]):
        for x in range(0, edges.shape[1]):
            if edges[y][x] == 0:
                continue
            # 更新累加器
            for theta in range(0, 180):
                rad = Util.rad(theta)
                rho = round(x * math.cos(rad) + y * math.sin(rad))
                accumMat[accumMat.shape[0] // 2 - rho][theta] += 1
    accumMatForShow = generateAccumMatForShow(accumMat)
    cv2.imwrite("Accum Mat for Line.jpg", accumMatForShow)
    pairList = pickPairs_line(accumMat, (img.shape[0], img.shape[1]), SuppressionMode.EIGHT_CONN)
    startPointList = []
    endPointList = []
    for i in range(0, len(pairList)):
        rho = pairList[i][0]
        theta = pairList[i][1]
        rad = Util.rad(theta)
        # 竖直
        if (abs(rad) < 0.1 or abs(rad - math.pi) < 0.1):
            startPointList.append((rho, 0))
            endPointList.append((rho, img.shape[0]))
            continue
        x1 = 0
        y1 = round((rho - x1 * math.cos(rad)) / math.sin(rad))
        startPointList.append((x1, y1))
        x2 = img.shape[1]
        y2 = round((rho - x2 * math.cos(rad)) / math.sin(rad))
        endPointList.append((x2, y2))
    for i in range(0, len(startPointList)):
        x1 = startPointList[i][0]
        y1 = startPointList[i][1]
        x2 = endPointList[i][0]
        y2 = endPointList[i][1]
        cv2.line(img, (x1, y1), (x2, y2), LINE_COLOR)
    cv2.imwrite("Img with Line.jpg", img)


def centerDetection(img: np.ndarray, edges: np.ndarray) -> None:
    accumMat = np.zeros(edges.shape, int)  # 圆心与边缘的距离大于图像本身尺度时舍弃
    # 计算x、y方向的微分
    sobelX = cv2.Sobel(edges, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(edges, cv2.CV_64F, 0, 1)
    cv2.imwrite("Sobel X.jpg", sobelX)
    cv2.imwrite("Sobel Y.jpg", sobelY)
    tanMat = np.zeros(edges.shape, float)
    for i in range(0, tanMat.shape[0]):
        for j in range(0, tanMat.shape[1]):
            if abs(sobelX[i][j]) < 1e-6 or abs(sobelY[i][j]) < 1e-6:
                continue
            else:
                # todo
                tanMat[i][j] = sobelY[i][j] / sobelX[i][j]
    # 计算累加器矩阵
    for y in range(0, edges.shape[0]):
        for x in range(0, edges.shape[1]):
            if edges[y][x] == 0:
                continue
            tan = tanMat[y][x]
            if tan == 0:
                continue
            for a in range(0, accumMat.shape[1]):
                b = round(a * tan - x * tan + y)
                if b < 0 or b >= accumMat.shape[0]:
                    continue
                accumMat[b][a] += 1
    accumMatForShow = generateAccumMatForShow(accumMat)
    cv2.imwrite("Accum Mat for Center.jpg", accumMatForShow)
    # 筛选圆心
    pairList = pickPairs_center(accumMat, (img.shape[0], img.shape[1]), SuppressionMode.EIGHT_CONN)
    # 标出圆心
    for i in range(0, len(pairList)):
        cv2.circle(img, pairList[i], 1, CENTER_COLOR, thickness=-1)  # thickness为负值时，填充圆形
    cv2.imwrite("Img with Center.jpg", img)
    cv2.imshow(WINDOW_NAME, img)
    cv2.waitKey(0)


def pickPairs_line(accumMat: np.ndarray, imgShape: tuple, mode: SuppressionMode) -> list:
    pairList = []
    T = min(imgShape[0], imgShape[1]) / 5  # 累加器的值大于该阈值时才入选
    for rho in range(0, accumMat.shape[0]):
        for theta in range(0, accumMat.shape[1]):
            if accumMat[rho][theta] > T:
                # 非极大值抑制
                if isLocalMaxima(accumMat, rho, theta, mode):
                    pairList.append((accumMat.shape[0] // 2 - rho, theta))
    return pairList


def pickPairs_center(accumMat: np.ndarray, imgShape: tuple, mode: SuppressionMode) -> list:
    pairList = []
    T = min(imgShape[0], imgShape[1]) / 64 * math.pi
    for a in range(0, accumMat.shape[1]):
        for b in range(0, accumMat.shape[0]):
            if accumMat[b][a] > T:
                if isLocalMaxima(accumMat, b, a, mode):
                    pairList.append((a, b))
    return pairList


def isLocalMaxima(accumMat: np.ndarray, i: int, j: int, mode: SuppressionMode) -> bool:
    this = accumMat[i][j]
    flag = True
    if mode == SuppressionMode.FOUR_CONN:
        if i > 0:
            if this < accumMat[i - 1][j]: flag = False
        if i < accumMat.shape[0] - 1:
            if this < accumMat[i + 1][j]: flag = False
        if j > 0:
            if this < accumMat[i][j - 1]: flag = False
        if j < accumMat.shape[1] - 1:
            if this < accumMat[i][j + 1]: flag = False
    else:
        if i > 0:
            if this < accumMat[i - 1][j]: flag = False
            if j > 0:
                if this < accumMat[i - 1][j - 1]: flag = False
            if j < accumMat.shape[1] - 1:
                if this < accumMat[i - 1][j + 1]: flag = False
        if i < accumMat.shape[0] - 1:
            if this < accumMat[i + 1][j]: flag = False
            if j > 0:
                if this < accumMat[i + 1][j - 1]: flag = False
            if j < accumMat.shape[1] - 1:
                if this < accumMat[i + 1][j + 1]: flag = False
        if j > 0:
            if this < accumMat[i][j - 1]: flag = False
        if j < accumMat.shape[1] - 1:
            if this < accumMat[i][j + 1]: flag = False
    return flag


def generateAccumMatForShow(accumMat: np.ndarray) -> np.ndarray:
    MAX = np.max(accumMat)
    accumMatForShow = np.zeros(accumMat.shape, np.uint8)
    for i in range(0, accumMat.shape[0]):
        for j in range(0, accumMat.shape[1]):
            accumMatForShow[i][j] = math.floor(accumMat[i][j] / MAX * 255)
    accumMatForShow = cv2.resize(accumMatForShow, (512, 512))
    return accumMatForShow


def main() -> None:
    img = cv2.imread(IMG_PATH)
    edges = cv2.Canny(img, CANNY_T[0], CANNY_T[1])  # edges为二维数组，元素全为0或255
    cv2.imwrite("Edges.jpg", edges)
    # lineDetection(img, edges)
    centerDetection(img, edges)


main()
