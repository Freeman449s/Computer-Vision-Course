import numpy as np
import cv2
import math

WINDOW_SIZE = 2 * 3 + 1  # 窗口尺寸，必须为非负奇数
SUPPRESS_WINDOW_SIZE = 2 * 3 + 1
R_T = 1E8
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
GRAY = (128, 128, 128)
USING_SOBEL = True
CAMERA_WINDOW_NAME = "Camera"


def computeDiffMatrices(gray: np.ndarray) -> np.ndarray:
    """
    为灰度图计算微分矩阵\n
    :param gray: 灰度图
    :return: 二维矩阵，二维矩阵中的每个元素都是一个微分矩阵
    """
    if USING_SOBEL:
        sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    diffMatrices = np.zeros(gray.shape, np.ndarray)
    for y in range(0, diffMatrices.shape[0]):
        for x in range(0, diffMatrices.shape[1]):
            M = diffMatrices[y][x] = np.zeros((2, 2), int)
            # 为实现方便，边缘点的差分矩阵赋为全0
            if y == diffMatrices.shape[0] - 1 or x == diffMatrices.shape[1] - 1:
                diffMatrices[y][x] = M
                continue
            if USING_SOBEL:
                Ix = round(sobelX[y][x] / 8)
                Iy = round(sobelY[y][x] / 8)
            else:
                Ix = int(gray[y][x + 1]) - int(gray[y][x])
                Iy = int(gray[y + 1][x]) - int(gray[y][x])
            M[0][0] = Ix * Ix
            M[0][1] = Ix * Iy
            M[1][0] = Ix * Iy
            M[1][1] = Iy * Iy
            diffMatrices[y][x] = M
    return diffMatrices


def computeMs(diffMatrices: np.ndarray) -> np.ndarray:
    """
    为图像中的每个点计算该点为中心窗口的M矩阵。边缘点将被忽略，计算结果为零矩阵\n
    :param diffMatrices: 图像的差分矩阵
    :return: 二维矩阵，矩阵的每个元素都是一个M矩阵
    """
    DELTA = WINDOW_SIZE // 2
    ZEROS = np.zeros((2, 2), int)
    Ms = np.zeros(diffMatrices.shape, np.ndarray)
    for y in range(0, Ms.shape[0]):
        for x in range(0, Ms.shape[1]):
            Ms[y][x] = ZEROS
            if y < DELTA or y >= Ms.shape[0] - DELTA or x < DELTA or x >= Ms.shape[1] - DELTA:
                continue
            else:
                if x == DELTA:
                    for j in range(-DELTA, DELTA + 1):
                        for i in range(-DELTA, DELTA + 1):
                            Ms[y][x] = np.add(Ms[y][x], diffMatrices[y + j][x + i])
                # x大于DELTA，利用左侧像素的M矩阵加速计算
                else:
                    Ms[y][x] = Ms[y][x - 1]
                    for j in range(-DELTA, DELTA + 1):
                        Ms[y][x] = np.add(Ms[y][x], diffMatrices[y + j][x + DELTA])
                        Ms[y][x] = np.add(Ms[y][x], -1 * diffMatrices[y + j][x - DELTA])
    return Ms


def computeLambdas(Ms: np.ndarray) -> np.ndarray:
    """
    为图像中的每一个点计算lambda值\n
    :param Ms: M矩阵形成的矩阵
    :return: 二维矩阵，矩阵中的每个元素是一个列表，从小到大存储了该点的M矩阵的lambda值
    """
    lambdas = np.zeros(Ms.shape, list)
    for y in range(0, Ms.shape[0]):
        for x in range(0, Ms.shape[1]):
            M = Ms[y][x]
            lambdas[y][x] = np.linalg.eigvals(M)
            lambdas[y][x].sort()
    return lambdas


def computeRs(Ms: np.ndarray) -> np.ndarray:
    """
    为图像中的每一个点计算R值\n
    :param Ms: M矩阵形成的矩阵
    :return: 图像中各点的R值形成的矩阵
    """
    K = 0.05
    Rs = np.zeros(Ms.shape, float)
    for y in range(0, Rs.shape[0] - 1):
        for x in range(0, Rs.shape[1] - 1):
            M = Ms[y][x]
            det = np.linalg.det(M)
            trace = np.trace(M)
            Rs[y][x] = det - K * trace * trace
    return Rs


def Harris(img: np.ndarray) -> np.ndarray:
    """
    对图像进行Harris角点检测，输出λmax图、λmin图、R图和最终标出角点的图\n
    :param img: 待检测图像
    :return: 标出Harris角点的图像
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diffMatrices = computeDiffMatrices(gray)
    Ms = computeMs(diffMatrices)
    # 计算并输出λ图
    lambdas = computeLambdas(Ms)
    lambdaMinima = np.zeros(lambdas.shape, float)
    lambdaMaxima = np.zeros(lambdas.shape, float)
    for y in range(0, lambdas.shape[0]):
        for x in range(0, lambdas.shape[1]):
            lambdaMinima[y][x] = lambdas[y][x][0]
            lambdaMaxima[y][x] = lambdas[y][x][1]
    maxOfLambdaMax = np.max(lambdaMaxima)
    maxOfLambdaMin = np.max(lambdaMinima)
    # 归一化
    for y in range(0, lambdas.shape[0]):
        for x in range(0, lambdas.shape[1]):
            lambdaMinima[y][x] = math.floor(lambdaMinima[y][x] / maxOfLambdaMin * 255)
            lambdaMaxima[y][x] = math.floor(lambdaMaxima[y][x] / maxOfLambdaMax * 255)
    cv2.imwrite("Lambda Maxima.jpg", lambdaMaxima)
    cv2.imwrite("Lambda Minima.jpg", lambdaMinima)
    # 计算R，输出R图，并在原图中标出角点
    Rs = computeRs(Ms)
    localMaximaPointList = findLocalMaxima(Rs)
    RImg = np.zeros(img.shape, np.uint8)
    harris = np.array(img)
    for y in range(0, Rs.shape[0]):
        for x in range(0, Rs.shape[1]):
            R = Rs[y][x]
            if abs(R) < R_T:
                for channel in range(0, 3):
                    RImg[y][x][channel] = GRAY[channel]
            elif R > 0:
                if (x, y) in localMaximaPointList:
                    cv2.circle(harris, (x, y), 1, GREEN, thickness=-1)
                for channel in range(0, 3):
                    RImg[y][x][channel] = RED[channel]
            else:
                for channel in range(0, 3):
                    RImg[y][x][channel] = GREEN[channel]
    cv2.imwrite("R Image.jpg", RImg)
    cv2.imwrite("Harris Corners.jpg", harris)
    return harris


def findLocalMaxima(Rs: np.ndarray) -> list:
    """
    对R进行非极大值抑制，避免特征点过度聚集\n
    :param Rs: R值矩阵
    :return: 局部极大值点列表
    """
    list = []
    for y in range(0, Rs.shape[0] - SUPPRESS_WINDOW_SIZE + 1, SUPPRESS_WINDOW_SIZE):
        for x in range(0, Rs.shape[1] - SUPPRESS_WINDOW_SIZE + 1, SUPPRESS_WINDOW_SIZE):
            max = -1E38
            maxPoint = (x, y)
            for j in range(0, SUPPRESS_WINDOW_SIZE):
                for i in range(0, SUPPRESS_WINDOW_SIZE):
                    if Rs[y + j][x + i] > max:
                        max = Rs[y + j][x + i]
                        maxPoint = (x + i, y + j)
            list.append(maxPoint)
    return list


def main() -> None:
    cap = cv2.VideoCapture(0)
    successful, frame = cap.read()
    while successful:
        cv2.imshow(CAMERA_WINDOW_NAME, frame)
        key = cv2.waitKey(40)
        if key == 32:
            cap.release()
            cv2.destroyWindow(CAMERA_WINDOW_NAME)
            break
        successful, frame = cap.read()
    print("Detecting Harris feature, stand by...")
    harris = Harris(frame)
    print("OK")
    cv2.imshow("Harris", harris)
    cv2.waitKey(0)


main()
