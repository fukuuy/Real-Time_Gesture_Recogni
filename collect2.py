import csv
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from add_datas import save_augmented_data
from normalizedata import DATA
from ui import GUI
import time

DATASET_FILE = "hand2_dataset.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

plt.figure(figsize=(5, 4))
ax = plt.subplot(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
hand_colors = ['red', 'blue']
lines = []
scatters = []
for color in hand_colors:
    hand_lines = [ax.plot([], [], [], c=color, linestyle='-')[0] for _ in range(21)]
    lines.append(hand_lines)
    scatter = ax.scatter([], [], [], c=color, marker='o')
    scatters.append(scatter)


def read_label(file_path=DATASET_FILE):
    labels = set()
    try:
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                label = row[-1]
                labels.add(label)
    except FileNotFoundError:
        print(f"文件未创建")
    return sorted(labels)


def save_data(features, hlabel):
    header = [f"{axis}0{i}" for i in range(0, 21) for axis in ["x", "y", "z"]] + [f"{axis}1{i}" for i in range(0, 21)
                                                                                  for axis in ["x", "y", "z"]] + [
                 "label"]
    data_row = np.append(features[0], features[1]).tolist() + [hlabel]
    with open(DATASET_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(header)
        writer.writerow(data_row)
    print(f"已保存（标签：{hlabel}）")


def update_plot(hands_data):
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
        (13, 17), (0, 17)
    ]
    all_x, all_y, all_z = [], [], []
    for hand_idx, hand_data in enumerate(hands_data):
        if hand_data:
            handxyz = np.array(hand_data).reshape(21, 3)
            handx = -handxyz[:, 0]
            handy = handxyz[:, 2]
            handz = -handxyz[:, 1]
            color = hand_colors[hand_idx]
            for j, (start, end) in enumerate(connections):
                lines[hand_idx][j].set_data([handx[start], handx[end]], [handy[start], handy[end]])
                lines[hand_idx][j].set_3d_properties([handz[start], handz[end]])
                lines[hand_idx][j].set_color(color)
            scatters[hand_idx]._offsets3d = (handx, handy, handz)
            all_x.extend(handx)
            all_y.extend(handy)
            all_z.extend(handz)
    if all_x:
        ax.set_xlim(min(all_x) - 0.1, max(all_x) + 0.1)
        ax.set_ylim(min(all_y) - 0.1, max(all_y) + 0.1)
        ax.set_zlim(min(all_z) - 0.1, max(all_z) + 0.1)

    plt.draw()
    plt.pause(0.001)


if __name__ == "__main__":
    collecting = False
    start_time = None
    plt.ion()
    cap = cv2.VideoCapture(0)
    gui = GUI(save_data)
    count = 0
    print(f"已记录标签{read_label()}")

    while cap.isOpened() and not gui.should_exit:
        success, image = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('TwoHands', cv2.flip(image, 1))

        hands_data = [[], []]
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                handxyz = []
                for landmark in hand_landmarks.landmark:
                    handxyz.extend([landmark.x, landmark.y, landmark.z])
                hand_label = 0 if handedness.classification[0].label == 'Left' else 1
                hands_data[hand_label] = handxyz
            hands_data = DATA.normalize_hands_data(hands_data)
            update_plot(hands_data)

        if gui.should_save:
            start_time = time.time()
            collecting = True
            total_count = gui.get_total_count()
            delay_time = gui.get_delay_time()
            add_num = gui.get_add_num()
            gui.reset_save_flag()

        if collecting:
            passed_time = time.time() - start_time
            label = int(gui.get_label())
            if passed_time >= delay_time and count < total_count:
                save_data(hands_data, label)
                count += 1
                print(f'已收集{count}条数据')
            elif count >= total_count:
                collecting = False
                count = 0
                print("收集完成")
                if add_num != 0:
                    save_augmented_data(label, add_num, DATASET_FILE)
                    print("数据增强完成")
            else:
                print(f'{int(delay_time - passed_time)}秒后开始记录')
        gui.update()

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close()
