import pandas as pd
import numpy as np


def add_random_noise(data, noise_scale=0.01):
    noise = np.random.normal(0, noise_scale, size=data.shape)
    new_data = data + noise
    if new_data.ndim == 3:
        new_data = new_data.reshape(new_data.shape[0], -1)
    return new_data


def rotate_data(data, angle_range=25):
    if data.ndim == 1:
        data = data.reshape(1, -1)
    num_points = data.shape[1] // 3
    data_3d = data.reshape(-1, num_points, 3)
    angle = np.deg2rad(np.random.uniform(-angle_range, angle_range))
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    # rotation_matrix = np.array([
    #     [cos_theta, -sin_theta, 0],
    #     [sin_theta, cos_theta, 0],
    #     [0, 0, 1]
    # ])  # z轴
    # rotation_matrix = np.array([
    #     [1, 0, 0],
    #     [0, cos_theta, -sin_theta],
    #     [0, sin_theta, cos_theta]
    # ])  # x轴
    rotation_matrix = np.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])  # y轴
    new_data_3d = np.zeros_like(data_3d)
    for sample_idx in range(data_3d.shape[0]):
        for point_idx in range(data_3d.shape[1]):
            point = data_3d[sample_idx, point_idx]
            new_point = np.dot(point, rotation_matrix.T)
            new_data_3d[sample_idx, point_idx] = new_point

    new_data = new_data_3d.reshape(data_3d.shape[0], -1)
    if new_data.shape[0] == 1:
        new_data = new_data.flatten()

    return new_data


def translate_data(data, translation_range=0.05):
    if data.ndim == 1:
        data = data.reshape(1, -1)
    translation = np.random.uniform(-translation_range, translation_range, size=(1, data.shape[1]))
    new_data = data + translation
    if new_data.ndim == 3:
        new_data = new_data.reshape(new_data.shape[0], -1)
    return new_data


def save_augmented_data(target_label, num_samples, file_path):
    data = pd.read_csv(file_path)
    label_data = data[data['label'] == target_label]
    labels = label_data['label'].values
    feature_data = label_data.drop(columns=['label'])
    original_samples = feature_data.values
    num_original_samples = len(original_samples)
    augmented_features = []
    for _ in range(num_samples):
        sample_index = np.random.randint(0, num_original_samples)
        sample = original_samples[sample_index].copy()
        sample = add_random_noise(sample)
        sample = rotate_data(sample)
        sample = sample.flatten()
        augmented_features.append(sample)
    augmented_df = pd.DataFrame(augmented_features, columns=feature_data.columns)
    selected_labels = np.random.choice(labels, num_samples)
    augmented_df['label'] = selected_labels
    with open(file_path, 'a', newline='') as f:
        augmented_df.to_csv(f, header=f.tell() == 0, index=False)


if __name__ == "__main__":
    FILE_PATH = 'hand_dataset.csv'
    label = 1
    add_num = 10
    save_augmented_data(label, add_num, FILE_PATH)
