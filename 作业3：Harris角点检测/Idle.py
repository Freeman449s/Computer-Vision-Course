"""
搁置的函数
"""
import numpy as np
from Exceptions import IllegalArgumentException

WINDOW_SIZE = 2 * 5 + 1


def computeLambdas(x: int, y: int, Ms: np.ndarray) -> tuple:
    """
    为给定点计算lambda值，返回最小值和最大值组成的元组\n
    :param x: 横坐标，应与边缘保持至少WINDOW_SIZE//2的距离，否则抛出异常
    :param y: 纵坐标，应与边缘保持至少WINDOW_SIZE//2的距离，否则抛出异常
    :param Ms: M矩阵形成的矩阵
    :return: 最小的lambda与最大的lambda组成的元组
    """
    DELTA = WINDOW_SIZE // 2
    # 边缘点忽略
    if x < DELTA or x > Ms.shape[1] - 1 - DELTA or y < DELTA or y > Ms.shape[1] - 1 - DELTA:
        raise IllegalArgumentException("(x,y) out of range.")
    M = Ms[y][x]
    # 计算中间矩阵
    MIDDLE_MAT = np.zeros((2, 2), int)
    for j in range(-DELTA, DELTA + 1):
        for i in range(-DELTA, DELTA + 1):
            MIDDLE_MAT = np.add(MIDDLE_MAT, M)
    # 计算E矩阵
    E = np.zeros((WINDOW_SIZE, WINDOW_SIZE), int)
    for j in range(0, WINDOW_SIZE):
        for i in range(0, WINDOW_SIZE):
            u = i - DELTA
            v = j - DELTA
            leftMat = np.array([u, v])
            rightMat = np.array([[u],
                                 [v]])
            E[j][i] = np.dot(np.dot(leftMat, MIDDLE_MAT), rightMat)
    lambdas = np.linalg.eigvals(E)
    lambdas.sort()
    return (lambdas[0], lambdas[-1])
