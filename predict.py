import cv2
import mediapipe as mp
import numpy as np
from dataprocess import DATA
import joblib
from collections import deque

model = joblib.load('gesture_model.pkl')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

prediction_history = deque(maxlen=5)

def predict_gesture(hand_data):

    normalized_data = DATA.normalize_hand_data1(hand_data)
    features = np.array(normalized_data).reshape(1, -1)

    prediction = model.predict(features)[0]
    probability = np.max(model.predict_proba(features))

    if probability > 0.7:
        prediction_history.append(prediction)

        if len(prediction_history) == prediction_history.maxlen:
            final_prediction = max(set(prediction_history), key=prediction_history.count)
            return final_prediction

    return None


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.extend([landmark.x, landmark.y, landmark.z])

                gesture_label = predict_gesture(hand_data)

                if gesture_label is not None:
                    cv2.putText(image, f"Gesture: {gesture_label}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Gesture Recognition', cv2.flip(image, 1))

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
