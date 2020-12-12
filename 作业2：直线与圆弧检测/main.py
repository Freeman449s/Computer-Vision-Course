from __future__ import annotations
import numpy as np
import cv2
import math
import Util
from enum import Enum
from Exception import IllegalArgumentException

IMG_PATH = "Highway Test.JPG"
WINDOW_NAME = "HW2"
CANNY_T = (320, 360)
LINE_COLOR = (255, 0, 0)
CENTER_COLOR = (0, 255, 0)
CIRCLE_COLOR = (0, 255, 0)
INF = 1E38


class Center():
    def __init__(self, pos: tuple):
        self.pos = pos
        self.distances = []
        self.radii = []

    def equals(self, other: Center):
        if self.pos[0] == other.pos[0] and self.pos[1] == other.pos[1]:
            return True
        else:
            return False


class SuppressionMode(Enum):
    FOUR_CONN = 1
    EIGHT_CONN = 2
    DOUBLE_EIGHT = 3  # 17*17
    DOUBLE_SIXTEEN = 4  # 33*33
    DOUBLE_THIRTY_SIX = 5  # 73*73


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
    cv2.imwrite("Img with Lines.jpg", img)


def circleDetection(img: np.ndarray, gray: np.ndarray, edges: np.ndarray) -> None:
    # 计算梯度方向
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=7)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=7)
    tanMat = np.zeros(gray.shape, float)
    for i in range(0, tanMat.shape[0]):
        for j in range(0, tanMat.shape[1]):
            if abs(sobelX[i][j]) < 1e-6 or abs(sobelY[i][j]) < 1e-6:
                continue
            else:
                tanMat[i][j] = sobelY[i][j] / sobelX[i][j]
    # 得到粗选后的圆心
    grossCenters = grossCenterDetection(img, gray, edges, tanMat)
    centers = []
    for i in range(0, len(grossCenters)):
        centers.append(Center(grossCenters[i]))
    # 筛选出圆弧上的点
    ARC_T = min(img.shape[0], img.shape[1]) / 128
    arcEdges = np.array(edges)
    for y in range(0, arcEdges.shape[0]):
        for x in range(0, arcEdges.shape[1]):
            if arcEdges[y][x] == 0:
                continue
            p = (x, y)
            nearestCenter = findNearestCenter(p, grossCenters)
            if not isOnArc(p, nearestCenter, tanMat, ARC_T):
                arcEdges[y][x] = 0
    cv2.imwrite("Arc Edges.jpg", arcEdges)
    # 令每个圆弧上的边缘点给最近的圆心投票
    for y in range(0, arcEdges.shape[0]):
        for x in range(0, arcEdges.shape[1]):
            if arcEdges[y][x] == 0:
                continue
            p = (x, y)
            nearestCenter = findNearestCenter(p, grossCenters)
            for i in range(0, len(centers)):
                if centers[i].equals(Center(nearestCenter)):
                    centers[i].distances.append(Util.distance(p, nearestCenter))
    # 筛选出票数大于阈值的中心
    N_VOTE_T = min(img.shape[0], img.shape[1]) / 16 * math.pi
    i = 0
    while i < len(centers):
        if len(centers[i].distances) < N_VOTE_T:
            centers.remove(centers[i])
            i -= 1
        i += 1
    # 标出筛选过的圆心
    for i in range(0, len(centers)):
        cv2.circle(img, centers[i].pos, 2, CENTER_COLOR, thickness=-1)
    cv2.imwrite("Img with Screened Centers.jpg", img)
    # 分析半径，绘制圆形
    RADIUS_DIST_T = min(img.shape[0], img.shape[1]) / 32
    for i in range(0, len(centers)):
        center = centers[i]
        analyzeRadii(center, RADIUS_DIST_T, N_VOTE_T)
        for j in range(0, len(center.radii)):
            cv2.circle(img, center.pos, round(center.radii[j]), CIRCLE_COLOR)
    cv2.imwrite("Img with Circles.jpg", img)
    cv2.imshow(WINDOW_NAME, img)
    cv2.waitKey(0)


def grossCenterDetection(img: np.ndarray, gray: np.ndarray, edges: np.ndarray, tanMat: np.ndarray) -> list:
    accumMat = np.zeros(gray.shape, int)  # 圆心与边缘的距离大于图像本身尺度时舍弃
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
    img_copy = np.array(img)
    for i in range(0, len(pairList)):
        cv2.circle(img_copy, pairList[i], 2, CENTER_COLOR, thickness=-1)  # thickness为负值时，填充圆形
    cv2.imwrite("Img with Gross Centers.jpg", img_copy)
    return pairList


def isOnArc(p: tuple, center: tuple, tanMat: np.ndarray, T: float) -> bool:
    """
    验证给定点p是否是以center为中心的圆上的点\n
    :param p:
    :param center:
    :param tanMat:
    :param T:
    :return:
    """
    px = p[0]
    py = p[1]
    cx = center[0]
    cy = center[1]
    tan = tanMat[py][px]
    C = py - tan * px
    dist = abs(tan * cx - cy + C) / math.sqrt(1 + tan * tan)
    return dist < T


def analyzeRadii(c: Center, DIST_T: float, VOTE_T: float) -> None:
    """
    分析围绕给定圆心的圆的半径\n
    :param c:
    :return:
    """
    c.distances.sort()
    dists = [c.distances[0]]
    for i in range(1, len(c.distances)):
        # 与第一个选中的距离差距不大，纳入此半径的计算
        if abs(c.distances[i] - dists[0]) < DIST_T:
            dists.append(c.distances[i])
        else:
            # 确保此半径确实是圆弧上的点产生的，只有票数大于阈值时才入选
            if len(dists) > VOTE_T:
                c.radii.append(np.mean(dists))
            dists = [c.distances[i]]
    if len(dists) > VOTE_T:
        c.radii.append(np.mean(dists))


def findNearestCenter(p: tuple, centers: list) -> tuple:
    if len(centers) < 1:
        raise IllegalArgumentException("Center list is empty.")
    minDist = INF
    nearest = centers[0]
    for i in range(0, len(centers)):
        dist = Util.distance(p, centers[i])
        if dist < minDist:
            minDist = dist
            nearest = centers[i]
    return nearest


def pickPairs_line(accumMat: np.ndarray, imgShape: tuple, mode: SuppressionMode) -> list:
    pairList = []
    T = min(imgShape[0], imgShape[1]) / 2.2  # 累加器的值大于该阈值时才入选
    for rho in range(0, accumMat.shape[0]):
        for theta in range(0, accumMat.shape[1]):
            if accumMat[rho][theta] > T:
                # 非极大值抑制
                if isLocalMaxima(accumMat, rho, theta, mode):
                    pairList.append((accumMat.shape[0] // 2 - rho, theta))
    return pairList


def pickPairs_center(accumMat: np.ndarray, imgShape: tuple, mode: SuppressionMode) -> list:
    # 过小点和非极大值点抑制
    tmpMat = np.array(accumMat)
    N_VOTE_T = min(imgShape[0], imgShape[1]) / 48 * math.pi
    for a in range(0, tmpMat.shape[1]):
        for b in range(0, tmpMat.shape[0]):
            if not tmpMat[b][a] > N_VOTE_T:
                tmpMat[b][a] = 0
            else:
                if not isLocalMaxima(tmpMat, b, a, mode):
                    tmpMat[b][a] = 0
    # 点按照票数的大小排序
    list = []
    for a in range(0, tmpMat.shape[1]):
        for b in range(0, tmpMat.shape[0]):
            if tmpMat[b][a] == 0:
                continue
            list.append((tmpMat[b][a], (a, b)))  # 列表的元素为元组，每个元组内包含点的值以及点的坐标
    pairList = []
    if len(list) < 1:
        return pairList
    list.sort(reverse=True)
    # 选取局部最大点。具体算法是先选中票数最多的点，在其之后距离小于阈值的点全部舍弃
    DISTANCE_T = min(imgShape[0], imgShape[1]) / 8
    pairList.append(list[0][1])
    for i in range(1, len(list)):
        p = list[i][1]
        if checkDistances(p, pairList, DISTANCE_T):
            pairList.append(p)
    return pairList


def checkDistances(p: tuple, pointList: list, T: float) -> bool:
    """
    检查传入的点是否与任何在列表中出现的距离过近\n
    :param p:
    :param pointList:
    :param T:
    :return:
    """
    for i in range(0, len(pointList)):
        if Util.distance(p, pointList[i]) < T:
            return False
    return True


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
    elif mode == SuppressionMode.EIGHT_CONN:
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
    # edges = cv2.GaussianBlur(edges, (3, 3), 1)
    cv2.imwrite("Edges.jpg", edges)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lineDetection(img, edges)
    circleDetection(img, gray, edges)


main()
