import numpy as np


class DATA:

    def normalize_hand_data1(hand_data):
        points = np.array(hand_data).reshape(21, 3)
        wrist = points[0]
        points -= wrist

        scale = np.linalg.norm(points[9] - points[0])
        points /= scale

        return points.flatten().tolist()

    def normalize_hand_data2(hand_data):  # 未完成
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

    @staticmethod
    def normalize_hand_data(hand_data):
        if not hand_data:
            return []
        points = np.array(hand_data).reshape(21, 3)
        wrist = points[0]
        points -= wrist

        scale = np.linalg.norm(points[9] - points[0])
        points /= scale

        return points.flatten().tolist()

    @staticmethod
    def normalize_hands_data(hands_data):
        left_hand = hands_data[0] if hands_data[0] else None
        right_hand = hands_data[1] if hands_data[1] else None
        if not (left_hand and right_hand):
            return [
                DATA.normalize_hand_data(left_hand) if left_hand else [],
                DATA.normalize_hand_data(right_hand) if right_hand else []
            ]
        left_points = np.array(left_hand).reshape(21, 3)
        right_points = np.array(right_hand).reshape(21, 3)
        all_points = np.vstack([left_points, right_points])
        center = np.mean(all_points, axis=0)
        left_points -= center
        right_points -= center

        left_palm_size = np.linalg.norm(left_points[9] - left_points[0])
        right_palm_size = np.linalg.norm(right_points[9] - right_points[0])
        interhand_dist = np.linalg.norm(left_points[0] - right_points[0])
        scale = (left_palm_size + right_palm_size + interhand_dist) / 3
        scale = max(scale, 1e-6)
        left_points /= scale
        right_points /= scale

        return [
            left_points.flatten().tolist(),
            right_points.flatten().tolist()
        ]
