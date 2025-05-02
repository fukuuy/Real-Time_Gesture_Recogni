import cv2
import mediapipe as mp
import numpy as np
from normalizedata import DATA
import joblib
from collections import deque


class GesturePredictor:
    def __init__(self):
        self.single_model = joblib.load('weight/hand/best_model.pkl')
        self.double_model = joblib.load('weight/hands/best_model.pkl')

        self.single_labels = {
            'one': [0],
            'two': [1, 7],
            'three': [2, 8],
            'four': [3],
            'five': [4],
            'six': [5, 6],
        }

        self.double_labels = {
            'love': [0],
            'fuck you': [1],
            'good': [2],
            'ten': [3],
        }

        self.single_id_to_name = self._create_id_to_name_mapping(self.single_labels)
        self.double_id_to_name = self._create_id_to_name_mapping(self.double_labels)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.single_history = deque(maxlen=5)
        self.double_history = deque(maxlen=5)
        self.hand_colors = [(0, 0, 255), (255, 0, 0)]

    def _create_id_to_name_mapping(self, label_dict):
        id_to_name = {}
        for name, ids in label_dict.items():
            for id_ in ids:
                id_to_name[id_] = name
        return id_to_name

    def _get_gesture_name(self, prediction, mode):
        if mode == 'single':
            return self.single_id_to_name.get(prediction, "unknown")
        else:
            return self.double_id_to_name.get(prediction, "unknown")

    def predict_single_gesture(self, hand_data):
        hand_data = DATA.normalize_hand_data1(hand_data)
        hand_data = hand_data[3:]
        features = np.array(hand_data).reshape(1, -1)
        prediction = self.single_model.predict(features)[0]
        probability = np.max(self.single_model.predict_proba(features))

        if probability > 0.4:
            self.single_history.append(prediction)
            if len(self.single_history) == self.single_history.maxlen:
                final_id = max(set(self.single_history), key=self.single_history.count)
                return self._get_gesture_name(final_id, 'single')
        return None

    def predict_double_gesture(self, hands_data):
        if not hands_data[0] or not hands_data[1]:
            return None
        normalized_data = DATA.normalize_hands_data(hands_data)
        left_hand = np.array(normalized_data[0]).reshape(21, 3)
        right_hand = np.array(normalized_data[1]).reshape(21, 3)
        features = np.concatenate([left_hand.flatten(), right_hand.flatten()]).reshape(1, -1)
        prediction = self.double_model.predict(features)[0]
        probability = np.max(self.double_model.predict_proba(features))

        if probability > 0.4:
            self.double_history.append(prediction)
            if len(self.double_history) == self.double_history.maxlen:
                final_id = max(set(self.double_history), key=self.double_history.count)
                return self._get_gesture_name(final_id, 'double')
        return None

    def process_frame(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        hands_data = [[], []]

        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)

            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                handxyz = []
                for landmark in hand_landmarks.landmark:
                    handxyz.extend([landmark.x, landmark.y, landmark.z])

                hand_label = 0 if handedness.classification[0].label == 'Left' else 1
                hands_data[hand_label] = handxyz

            if hand_count == 1:
                hand_data = hands_data[0] if hands_data[0] else hands_data[1]
                hand_data = DATA.normalize_hand_data1(hand_data)
                gesture_name = self.predict_single_gesture(hand_data)

                if gesture_name is not None:
                    cv2.putText(image, gesture_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif hand_count == 2:
                gesture_name = self.predict_double_gesture(hands_data)
                if gesture_name is not None:
                    cv2.putText(image, gesture_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return image


if __name__ == "__main__":
    predictor = GesturePredictor()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = predictor.process_frame(image)
        cv2.imshow('Gesture Recognition', image)

        if cv2.waitKey(1) & 0xFF == 27:  # ESCÍË³ö
            break

    cap.release()
    cv2.destroyAllWindows()
