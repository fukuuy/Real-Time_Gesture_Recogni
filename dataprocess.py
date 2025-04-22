import numpy as np
from scipy.spatial.transform import Rotation


class DATA:

    def normalize_hand_data1(hand_data):  # 无计算旋转矩阵
        points = np.array(hand_data).reshape(21, 3)

        wrist = points[0]
        points -= wrist

        scale = np.linalg.norm(points[9] - points[0])
        points /= scale

        return points.flatten().tolist()

    def normalize_hand_data2(hand_data):
        points = np.array(hand_data).reshape(21, 3)

        wrist = points[0]
        points -= wrist
        scale = np.linalg.norm(points[9] - points[0])
        points /= scale

        v1 = points[9] - points[0]
        v2 = points[13] - points[0]
        normal1 = np.cross(v1, v2)
        normal1 /= np.linalg.norm(normal1)
        normal2 = np.array([0, 1, 0])
        dot_product = np.dot(normal1, normal2)
        cos_angle = np.abs(dot_product)
        angle = np.arccos(cos_angle)

        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        points = np.dot(points, rotation_matrix.T)

        return points.flatten().tolist()
