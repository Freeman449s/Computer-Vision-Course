import numpy as np
import os
import cv2
from Errors import IllegalArgumentError
import matplotlib.pyplot as plt
import math

MODEL_FILE_PATH = "model.npy"
AVG_FILE_PATH = "avg.npy"
FACE_LIB_PATH = "ORL Library"
MY_FACE_LIB_PATH = "My Face Lib"
TEST_FACE_PATH = "Test Face.pgm"
INF = 1E38


def train(faces: list, energyRatio: float) -> tuple:
    """
    训练模型\n
    :param faces: 人脸数据
    :param energyRatio: 能量比，用于确定所需基向量的个数
    :param modelFilePath: 用于储存模型的文件的路径
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
    eigVals = np.real(eigVals)  # 实数化
    eigVecs = np.real(eigVecs)
    sortEigVecs(eigVals, eigVecs, N_PIXEL)
    # 计算所需基的个数
    k = 0
    eigValSum = 0
    while eigValSum < np.sum(eigVals) * energyRatio:
        eigValSum += eigVals[-1 - k]
        k += 1
    baseVecs = np.zeros((N_PIXEL, k), float)
    for i in range(0, k):
        baseVecs[:, i] = eigVecs[:, -1 - i]
    # Step5: 写入模型
    # with open(modelFilePath, "w") as file:
    #     # 先写入向量维数与向量个数
    #     file.write(str(N_PIXEL) + " " + str(k) + "\n")
    #     # 写入平均脸
    #     for row in range(0, N_PIXEL):
    #         file.write(avg[row][0])
    #         if row < N_PIXEL - 1:
    #             file.write(" ")
    #     file.write("\n")
    #     # 一个向量作为一行写入
    #     for col in range(0, k):
    #         for row in range(0, N_PIXEL):
    #             file.write(str(baseVecs[row][col]))
    #             if row < N_PIXEL - 1:
    #                 file.write(" ")
    #         file.write("\n")
    np.save(MODEL_FILE_PATH, baseVecs)
    np.save(AVG_FILE_PATH, avg)
    return (avg, baseVecs)


def showAvgAndTopTen(baseVecs: np.ndarray, WIDTH: int, HEIGHT: int) -> None:
    """
    显示前十张特征脸\n
    :param baseVecs: 基向量矩阵
    :param WIDTH: 人脸图的宽度
    :param HEIGHT: 人脸图的高度
    :return: 无返回值
    """
    N_PIXEL = baseVecs.shape[0]
    for i in range(0, 10):
        face = np.array(baseVecs[:, i])
        face = np.resize(face, (HEIGHT, WIDTH))
        # face = normalize(face)
        plt.subplot(2, 5, i + 1)  # 行数，列数，序号
        plt.imshow(face, cmap="gray")  # 灰度显示
        plt.xticks([])  # 去除刻度
        plt.yticks([])
        # vec = np.array(baseVecs[:, i])
        # vec = np.resize(vec, (N_PIXEL, 1))
        # topTen = np.add(topTen, vec)
    plt.show()
    # topTen = np.resize(topTen, (HEIGHT, WIDTH))
    # topTenUByte = normalize(topTen)
    # cv2.imwrite("Top Ten Stitched.jpg", topTenUByte)
    # cv2.imshow("Top Ten", topTenUByte)
    # cv2.waitKey(0)


def showAvg(avg: np.ndarray, WIDTH: int, HEIGHT: int) -> None:
    """
    展示平均脸\n
    :param avg: 平均脸（向量形式）
    :param WIDTH: 人脸图的宽度
    :param HEIGHT: 人脸图的高度
    :return: 无返回值
    """
    avgMat = np.resize(avg, (HEIGHT, WIDTH))
    plt.imshow(avgMat, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def reconstruct(face: np.ndarray, avg: np.ndarray, baseVecs: np.ndarray, N_PCS: int) -> np.ndarray:
    """
    将传入人脸变换到特征脸空间再重构\n
    :param face: 待重构人脸
    :param avg: 平均脸
    :param baseVecs: 基向量矩阵
    :param WIDTH: 重构图像的宽度
    :param HEIGHT: 重构图像的高度
    :param N_PCS: 重构使用的主元个数
    :return: 重构的人脸（已映射到[0,255]）
    """
    WIDTH = face.shape[1]
    HEIGHT = face.shape[0]
    faceCoord = computeCoord(face, avg, baseVecs, N_PCS)
    # k = baseVecs.shape[1]
    N_PIXEL = WIDTH * HEIGHT
    reconFace = np.zeros((N_PIXEL, 1), float)
    for i in range(0, N_PCS):
        vec = baseVecs[:, i]
        vec = np.resize(vec, (N_PIXEL, 1))
        reconFace = np.add(np.float64(reconFace), np.float64(faceCoord[i] * vec))
    reconFace = np.add(reconFace, avg)
    reconFace = np.resize(reconFace, (HEIGHT, WIDTH))
    reconFace = normalize(reconFace)
    windowName = "Reconstructed - " + str(N_PCS) + "PCs"
    cv2.imwrite(windowName + ".jpg", reconFace)
    cv2.imshow(windowName, reconFace)
    cv2.waitKey(0)
    return reconFace


def normalize(gray: np.ndarray) -> np.ndarray:
    """
    将灰度图映射到[0,255]范围\n
    :param gray: 浮点类型的灰度图
    :return: 映射完成后的灰度图
    """
    ubyteGray = np.zeros(gray.shape, np.uint8)
    max = np.max(gray)
    for y in range(0, gray.shape[0]):
        for x in range(0, gray.shape[1]):
            ubyteGray[y][x] = np.uint8(gray[y][x] / max * 255)
    return ubyteGray


def computeCoord(faceMat: np.ndarray, avg: np.ndarray, baseVecs: np.ndarray, LENGTH: int) -> np.ndarray:
    """
    计算人脸在给定基向量下的坐标\n
    :param faceMat: 二维矩阵形式的人脸
    :param avg: 向量形式的平均脸
    :param baseVecs: 基向量矩阵
    :param LENGTH: 返回的坐标的维数
    :return: 人脸在给定基向量下的坐标
    """
    # 矩阵转向量
    N_PIXEL = avg.shape[0]
    face = np.resize(faceMat, (N_PIXEL, 1))
    # 计算坐标
    k = baseVecs.shape[1]  # 特征向量的个数
    coord = np.zeros((k, 1))
    diffVec = np.subtract(face, avg)
    for i in range(0, k):
        baseVec = np.array(baseVecs[:, i])
        coord[i][0] = np.matmul(baseVec, diffVec)
    coord = coord[0:LENGTH, 0]
    coord = np.resize(coord, (LENGTH, 1))
    return coord


def vecCos(vecA: np.ndarray, vecB: np.ndarray) -> float:
    """
    计算同维向量的夹角余弦\n
    :param vecA: 向量A
    :param vecB: 向量B
    :return: 两向量的夹角余弦
    """
    if vecA.shape[0] != vecB.shape[0] or vecA.shape[1] != vecB.shape[1]:
        raise IllegalArgumentError("vecA and vecB's shapes are not the same.")
    return np.sum(vecA * vecB) / np.linalg.norm(vecA) / np.linalg.norm(vecB)


def findMostSimilar(face: np.ndarray, faces: list, avg: np.ndarray, baseVecs: np.ndarray, N_PCS: int) -> np.ndarray:
    """
    在人脸库中寻找与给定人脸最相似的人脸\n
    :param face: 输入人脸
    :param faces: 人脸库
    :param avg: 平均脸
    :param baseVecs: 基向量矩阵
    :param N_PCS: 使用的坐标的维数
    :return: 最相似人脸的灰度图
    """
    mostSimilar = faces[0]
    faceCoord = computeCoord(face, avg, baseVecs, N_PCS)
    libFaceCoord = computeCoord(mostSimilar, avg, baseVecs, N_PCS)
    maxSim = vecCos(faceCoord, libFaceCoord)
    for i in range(1, len(faces)):
        libFace = faces[i]
        libFaceCoord = computeCoord(libFace, avg, baseVecs, N_PCS)
        sim = vecCos(faceCoord, libFaceCoord)
        if sim > maxSim:
            mostSimilar = libFace
            maxSim = sim
    return mostSimilar


def identify(face: np.ndarray, faces: list, avg: np.ndarray, baseVecs: np.ndarray, N_PCS: int) -> None:
    """
    查找并显示最相似的人脸\n
    :param face: 待识别人脸
    :param faces: 人脸库
    :param avg: 平均脸
    :param baseVecs: 基向量
    :param N_PCS: 使用的坐标的维数
    :return: 无返回值
    """
    mostSimilar = findMostSimilar(face, faces, avg, baseVecs, N_PCS)
    # 显示输入人脸
    plt.subplot(1, 3, 1)
    plt.imshow(face, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.title("Input Face")

    mixed = np.add(np.int32(mostSimilar), np.int32(face))
    mixed = normalize(mixed)
    # 显示混合人脸
    plt.subplot(1, 3, 2)
    plt.imshow(mixed, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.title("Mixed")
    # 显示最相似人脸
    plt.subplot(1, 3, 3)
    plt.imshow(mostSimilar, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.title("Most Similar")
    plt.show()


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
        keyVec[:, 0] = eigVecs[:, i]
        j = i - 1
        while j >= 0:
            if eigVals[j] > keyVal:
                eigVals[j + 1] = eigVals[j]
                eigVecs[:, j + 1] = eigVecs[:, j]
            else:
                break
            j -= 1
        eigVals[j + 1] = keyVal
        eigVecs[:, j + 1] = keyVec[:, 0]


def readFaces() -> tuple:
    """
    从本地读取人脸图像\n
    :return: 库人脸的列表以及自己人脸的列表组成的元组
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
    myFaces = []
    for root, dirs, files in os.walk(MY_FACE_LIB_PATH):
        for file in files:
            filePath = os.path.join(root, file)
            face = cv2.imread(filePath)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            myFaces.append(face)
    return (faces, myFaces)


def readFace(filePath: str) -> np.ndarray:
    """
    依据传入的文件路径读取人脸\n
    :param filePath: 文件路径
    :return: 读取的人脸
    """
    face = cv2.imread(filePath)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return face


def importModel() -> tuple:
    """
    从文件中读取模型\n
    :param modelFilePath: 模型文件路径
    :return: 平均脸和基向量矩阵
    """
    # with open(modelFilePath, "r") as file:
    #     # 先读取向量维数和个数
    #     infoLine = file.readline()
    #     elements = infoLine.split(" ")
    #     N_PIXEL = float(elements[0])
    #     k = int(elements[1])
    #     # 读取平均脸
    #     avgLine = file.readline()
    #     elements = avgLine.split(" ")
    #     avg = np.zeros((N_PIXEL, 1), float)
    #     for row in range(0, len(elements)):
    #         avg[row][0] = float(elements[row])
    #     # 读取特征向量
    #     baseVecs = np.zeros((N_PIXEL, k), float)
    #     col = 0
    #     for line in file:
    #         elements = line.split(" ")
    #         for row in range(0, len(elements)):
    #             baseVecs[row][col] = float(elements[row])
    #         col += 1
    baseVecs = np.load(MODEL_FILE_PATH)
    avg = np.load(AVG_FILE_PATH)
    return (avg, baseVecs)


def main() -> None:
    orlFaces, myFaces = readFaces()
    allFaces = orlFaces[:]
    for i in range(0, len(myFaces)):
        allFaces.append(myFaces[i])
    train(orlFaces, 0.99)
    # testFace = readFace("My Test Face.jpg")
    # avg, baseVecs = importModel()
    # FACE_WIDTH = faces[0].shape[1]
    # FACE_HEIGHT = faces[0].shape[0]
    # k = baseVecs.shape[1]


main()
