import numpy as np
import cv2
import math
from PIL import Image

IMG_PATH = "Test Image.jpg"
WINDOW_NAME = "HW2"
CANNY_T = (360, 400)


def lineDetection(img: np.ndarray, edges: np.ndarray) -> None:
    accumMat = np.zeros(((edges.shape[0] + edges.shape[1]) * 2 + 1, 180), int)
    for y in range(0, edges.shape[0]):
        for x in range(0, edges.shape[1]):
            if edges[y][x] == 0:
                continue
            # 更新累加器
            for theta in range(0, 180):
                rad = theta / 360 * 2 * math.pi
                rho = round(x * math.cos(rad) + y * math.sin(rad))
                accumMat[accumMat.shape[0] // 2 - rho][theta] += 1
    accumMatForShow = generateAccumMatForShow(accumMat)
    cv2.imshow(WINDOW_NAME, accumMatForShow)
    cv2.waitKey(0)


def generateAccumMatForShow(accumMat: np.ndarray) -> np.ndarray:
    MAX = np.max(accumMat)
    accumMatForShow = np.zeros(accumMat.shape, np.float32)
    for i in range(0, accumMat.shape[0]):
        for j in range(0, accumMat.shape[1]):
            accumMatForShow[i][j] = accumMat[i][j] / MAX
    accumMatForShow = cv2.resize(accumMatForShow, (512, 512))
    return accumMatForShow


def main() -> None:
    img = cv2.imread(IMG_PATH)
    edges = cv2.Canny(img, CANNY_T[0], CANNY_T[1])  # edges为二维数组，元素全为0或255
    lineDetection(img, edges)


main()
