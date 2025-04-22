import csv
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from dataprocess import DATA
from ui import GUI

DATASET_FILE = "hand_dataset.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

fig = plt.figure(figsize=(5, 4))
ax = plt.subplot(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
lines = []
scatters = []
hand_lines = [ax.plot([], [], [], c='r', linestyle='-')[0] for _ in range(21)]
lines.append(hand_lines)
scatter = ax.scatter([], [], [], c='b', marker='o')
scatters.append(scatter)


def update_plot(hand_data):
    for hand_lines in lines:
        for line in hand_lines:
            line.set_data([], [])
            line.set_3d_properties([])

    for scatter in scatters:
        scatter._offsets3d = (np.array([]), np.array([]), np.array([]))

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20),
        (13, 17), (0, 17),
    ]

    handx = -1 * np.array(hand_data[0::3])
    handy = np.array(hand_data[2::3])
    handz = -1 * np.array(hand_data[1::3])

    for j, (start, end) in enumerate(connections):
        lines[0][j].set_data([handx[start], handx[end]], [handy[start], handy[end]])
        lines[0][j].set_3d_properties([handz[start], handz[end]])
    scatters[0]._offsets3d = (handx, handy, handz)

    if hand_data:
        ax.set_xlim(min(handx) - 0.1, max(handx) + 0.1)
        ax.set_ylim(min(handy) - 0.1, max(handy) + 0.1)
        ax.set_zlim(min(handz) - 0.1, max(handz) + 0.1)

    plt.draw()
    plt.pause(0.001)


def save_gesture_data(features, hlabel):
    header = [f"{axis}{i}" for i in range(1, 21) for axis in ["x", "y", "z"]] + ["label"]
    data_row = np.append(features[3:], [hlabel]).tolist()

    with open(DATASET_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(header)
        writer.writerow(data_row)
    print(f"已保存（标签：{hlabel}）")


def main():
    plt.ion()
    cap = cv2.VideoCapture(0)

    gui = GUI(save_gesture_data)

    while cap.isOpened() and not gui.should_exit:
        success, image = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('OneHand', cv2.flip(image, 1))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                handxyz = []
                for landmark in hand_landmarks.landmark:
                    handxyz.extend([landmark.x, landmark.y, landmark.z])
                handxyz = np.array(handxyz, dtype=np.float32)
                handxyz = DATA.normalize_hand_data1(handxyz)
                update_plot(handxyz)

                if gui.should_save:
                    label = gui.get_label()
                    save_gesture_data(handxyz, int(label))
                    gui.reset_save_flag()

        gui.update()

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close()


if __name__ == "__main__":
    main()
