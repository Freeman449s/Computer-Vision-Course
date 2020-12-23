import numpy as np


def train(faces: np.ndarray) -> None:
    # Step1: 二维矩阵转向量
    faceLists = []
    N_FACES = faces.shape[0]
    for i in range(0, N_FACES):
        faceList = []
        face = faces[i]  # faces为三维矩阵，faces[i]是第i张人脸
        for y in range(0, face.shape[0]):
            for x in range(0, face.shape[1]):
                faceList.append(face[y][x])
        faceLists.append(faceList)
    # Step2: 求平均脸
    N_PIXEL = faces.shape[1] * faces.shape[2]
    avg = []
    for i in range(0, len(faceLists[0])):
        # 求平均脸第i维的值
        sum = 0
        for j in range(0, len(faceLists)):
            sum += faceLists[j][i]
        sum /= N_PIXEL
        avg.append(sum)
    # Step3: 求协方差阵
    C = np.zeros((N_PIXEL, N_PIXEL))
    for i in range(0, N_FACES):
        # 求人脸与平均脸之差
        diffList = []
        for j in range(0, N_PIXEL):
            diffList.append(faceLists[i][j] - avg[j])
        diffVecT = np.array(diffList)  # 差向量的转置，尺寸1*N_PIXEL
        diffVecT.resize((1, N_PIXEL))
        diffVec = np.array(diffVecT)
        diffVec.resize((N_PIXEL, 1))  # 尺寸N_PIXEL*1
        C = np.add(C, np.matmul(diffVec, diffVecT))
    #todo 求特征向量


def main() -> None:
    pass
