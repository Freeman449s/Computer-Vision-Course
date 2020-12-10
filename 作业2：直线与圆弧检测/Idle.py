"""
被闲置的函数
"""
import numpy as np
from main import SuppressionMode
from main import isLocalMaxima


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
