"""
被闲置的函数
"""
import numpy as np
from main import SuppressionMode
from main import isLocalMaxima
from main import generateAccumMatForShow


def centerEnhance(m: np.ndarray, mode: SuppressionMode) -> np.ndarray:
    T = np.max(m) / 10
    ret = np.array(m, int)
    for i in range(0, ret.shape[0]):
        for j in range(0, ret.shape[1]):
            if ret[i][j] < T or not isLocalMaxima(ret, i, j, mode):
                ret[i][j] = 0
            else:
                ret[i][j] *= 2
    return ret


def generateAccumMatForShow_circle(accumMat: np.array, pairList: list) -> np.ndarray:
    m = np.zeros(accumMat.shape, int)
    for b in range(accumMat.shape[0]):
        for a in range(0, accumMat.shape[1]):
            if (a, b) not in pairList:
                continue
            m[b][a] = accumMat[b][a]
    return generateAccumMatForShow(m)
