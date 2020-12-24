import numpy as np
import os
import cv2
from Errors import IllegalArgumentError

MODEL_FILE_PATH = "model.txt"
FACE_LIB_PATH = "ORL Library"
INF = 1E38


def train(faces: list, energyRatio: float) -> tuple:
    """
    训练模型\n
    :param faces: 人脸数据
    :param energyRatio: 能量比，用于确定所需基向量的个数
    :return: 平均脸和基向量组成的矩阵
    """
    # Step1: 二维矩阵转向量
    faceVecs = []
    N_FACES = len(faces)
    WIDTH = faces[0].shape[1]
    HEIGHT = faces[0].shape[0]
    N_PIXEL = WIDTH * HEIGHT
    for i in range(0, N_FACES):
        face = faces[i]  # faces为二维矩阵组成的列表，faces[i]是第i张人脸
        faceVec = np.array(face)
        faceVec.resize((N_PIXEL, 1))
        faceVecs.append(faceVec)
    # Step2: 求平均脸
    avg = np.zeros((N_PIXEL, 1), float)
    for i in range(0, N_PIXEL):
        # 求平均脸第i维的值
        sum = 0.0
        for j in range(0, N_FACES):
            sum += faceVecs[j][i][0]
        sum /= N_FACES
        avg[i][0] = sum
    # Step3: 求协方差阵
    C = np.zeros((N_PIXEL, N_PIXEL), float)
    for i in range(0, N_FACES):
        # 求人脸与平均脸之差
        diffVec = np.subtract(faceVecs[i], avg)
        diffVecT = np.array(diffVec)  # 差向量的转置，尺寸1*N_PIXEL
        diffVecT.resize((1, N_PIXEL))
        C = np.add(C, np.matmul(diffVec, diffVecT))
    C /= N_FACES
    # Step4: 求出并筛选特征向量
    eigVals, eigVecs = np.linalg.eig(C)
    sortEigVecs(eigVals, eigVecs, N_PIXEL)
    # 计算所需基的个数
    k = 0
    eigValSum = 0
    while eigValSum < np.sum(eigVals) * energyRatio:
        eigValSum += eigVals[k]
        k += 1
    baseVecs = np.zeros((N_PIXEL, k), float)
    for i in range(0, k):
        baseVecs[:, i] = eigVecs[:, -1 - i]
    # Step5: 写入模型
    with open(MODEL_FILE_PATH, "w")  as file:
        # 一个向量作为一行写入
        for col in range(0, k):
            for row in range(0, N_PIXEL):
                file.write(str(baseVecs[row][col]))
                if row < N_PIXEL - 1:
                    file.write(" ")
            file.write("\n")
    # 拼凑前10张特征脸并显示
    topTen = np.zeros((N_PIXEL, 1))
    for i in range(0, 10):
        topTen = np.add(topTen, baseVecs[:, i])
    topTen.resize((HEIGHT, WIDTH))
    topTenUByte = float2ubyte(topTen)
    cv2.imwrite("Top Ten Stitched.jpg", topTenUByte)
    cv2.imshow("Top Ten", topTenUByte)
    cv2.waitKey(0)
    return (avg, baseVecs)


def reconstruct(face: np.ndarray, avg: np.ndarray, baseVecs: np.ndarray, WIDTH: int, HEIGHT: int) -> np.ndarray:
    """
    将传入人脸变换到特征脸空间再重构\n
    :param face: 待重构人脸
    :param avg: 平均脸
    :param baseVecs: 基向量矩阵
    :param WIDTH: 重构图像的宽度
    :param HEIGHT: 重构图像的高度
    :return: 重构的人脸（已映射到[0,255]）
    """
    faceCoord = computeCoord(face, avg, baseVecs)
    k = baseVecs.shape[1]
    N_PIXEL = WIDTH * HEIGHT
    reconFace = np.zeros((N_PIXEL, 1))
    for i in range(0, k):
        reconFace = np.add(reconFace, faceCoord[i] * baseVecs[:, i])
    reconFace.resize((HEIGHT, WIDTH))
    reconFace = float2ubyte(reconFace)
    cv2.imwrite("Reconstructed.jpg", reconFace)
    cv2.imshow("Reconstructed", reconFace)
    cv2.waitKey(0)
    return reconFace


def float2ubyte(gray: np.ndarray) -> np.ndarray:
    """
    将灰度图映射到[0,255]范围\n
    :param gray: 浮点类型的灰度图
    :return: 映射完成后的灰度图
    """
    ubyteGray = np.array(gray.shape, np.uint8)
    max = np.max(gray)
    for y in range(0, gray.shape[0]):
        for x in range(0, gray.shape[1]):
            ubyteGray[y][x] = round(gray[y][x] / max * 255)
    return ubyteGray


def computeCoord(faceMat: np.ndarray, avg: np.ndarray, baseVecs: np.ndarray) -> np.ndarray:
    """
    计算人脸在给定基向量下的坐标\n
    :param faceMat: 二维矩阵形式的人脸
    :param avg: 向量形式的平均脸
    :param baseVecs: 基向量矩阵
    :return: 人脸在给定基向量下的坐标
    """
    # 矩阵转向量
    face = np.array(faceMat[0, :])
    for row in range(1, faceMat.shape[0]):
        np.append(face, faceMat[row, :], 1)
    face.resize((face.shape[1], 1))
    # 计算坐标
    k = baseVecs.shape[1]  # 特征向量的个数
    coord = np.zeros((k, 1))
    diffVec = np.subtract(face, avg)
    for i in range(0, k):
        baseVec = np.array(baseVecs[:, i])
        coord[i][0] = np.matmul(baseVec, diffVec)
    return coord


def vecCos(vecA: np.ndarray, vecB: np.ndarray) -> float:
    if vecA.shape[0] != vecB.shape[0] or vecA.shape[1] != vecB.shape[1]:
        raise IllegalArgumentError("vecA and vecB's shapes are not the same.")
    return (vecA * vecB) / np.linalg.norm(vecA) / np.linalg.norm(vecB)


def findMostSimilar(face: np.ndarray, faces: list, avg: np.ndarray, baseVecs: np.ndarray) -> np.ndarray:
    """
    在人脸库中寻找与给定人脸最相似的人脸\n
    :param face: 输入人脸
    :param faces: 人脸库
    :param avg: 平均脸
    :param baseVecs: 基向量矩阵
    :return: 最相似人脸的灰度图
    """
    mostSimilar = faces[0]
    faceCoord = computeCoord(face, avg, baseVecs)
    libFaceCoord = computeCoord(mostSimilar, avg, baseVecs)
    maxSim = vecCos(faceCoord, libFaceCoord)
    for i in range(1, len(faces)):
        libFace = faces[i]
        libFaceCoord = computeCoord(libFace, avg, baseVecs)
        sim = vecCos(faceCoord, libFaceCoord)
        if sim > maxSim:
            mostSimilar = libFace
            maxSim = sim
    return mostSimilar


def sortEigVecs(eigVals: np.ndarray, eigVecs: np.ndarray, LENGTH: int) -> None:
    """
    将特征值和特征向量依据特征值的大小从小到大排序\n
    :param eigVals: 特征值
    :param eigVecs: 特征向量
    :param LENGTH: 特征向量的长度
    :return: 无返回值
    """
    keyVec = np.zeros((LENGTH, 1), float)
    for i in range(1, len(eigVals)):
        keyVal = eigVals[i]
        keyVec[:, :] = eigVecs[:, i]
        j = i - 1
        while j >= 0:
            if eigVals[j] > keyVal:
                eigVals[j + 1] = eigVals[j]
                eigVecs[:, j + 1] = eigVals[:, j]
            else:
                break
            j -= 1
        eigVals[j + 1] = keyVal
        eigVecs[:, j + 1] = keyVec


def readFaces() -> list:
    """
    从本地读取人脸图像\n
    :return: 人脸图像组成的列表
    """
    faces = []
    for outerRoot, outerDirs, outerFiles in os.walk(FACE_LIB_PATH):
        for outerDir in outerDirs:
            outerDir = os.path.join(outerRoot, outerDir)
            for innerRoot, innerDirs, innerFiles in os.walk(outerDir):
                for file in innerFiles:
                    filePath = os.path.join(innerRoot, file)
                    face = cv2.imread(filePath)
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # 读入的人脸是三通道灰度图
                    faces.append(face)
    return faces


def main() -> None:
    faces = readFaces()
    train(faces, 0.95)


main()
