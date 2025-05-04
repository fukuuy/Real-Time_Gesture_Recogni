# Gesture Recognition Project

## I. Project Overview
This project is a Python-based gesture recognition system that uses the Mediapipe library for hand keypoint detection and combines multiple machine learning models (such as Random Forest, Support Vector Machine, K-Nearest Neighbors, XGBoost, and Multi-Layer Perceptron) to achieve gesture classification and recognition. The project supports single-gesture and double-gesture recognition, and provides functions such as data collection, data augmentation, model training, and real-time recognition.

## II. Project Structure
```
Gestrue_Recognition/
├── add_datas.py            # Data augmentation script
├── collect.py              # Single-gesture data collection script
├── collect2.py             # Double-gesture data collection script
├── normalizedata.py        # Data normalization processing script
├── recognition.py          # Real-time recognition script for single and double gestures
├── recognition1.py         # Real-time single-gesture recognition script
├── recognition2.py         # Real-time double-gesture recognition script
├── requirements.txt        # Project dependency library file
├── train.py                # Model training script
├── ui.py                   # Data collection interface script
├── read_csv.py             # Script to read CSV files and visualize gesture data
└── data/                   # Dataset storage directory
    ├── hand_dataset.csv    # Single-gesture dataset
    └── hand2_dataset.csv   # Double-gesture dataset
└── weight/                 # Model weight and label mapping storage directory
    ├── hand/               # Single-gesture model-related files
    │   ├── best_model.pkl  # Best single-gesture model
    │   └── label_encoder.pkl # Single-gesture label mapping
    └── hands/              # Double-gesture model-related files
        ├── best_model.pkl  # Best double-gesture model
        └── label_encoder.pkl # Double-gesture label mapping
```

## III. Environment Configuration
```bash
pip install -r requirements.txt
```

## IV. Usage Instructions

### 1. Data Collection
#### Single-gesture data collection
```bash
python collect.py
```
After running this script, a graphical interface will pop up. You can set the gesture label, the number of data to be collected, the delay time, and the number of augmented data. Click the "Start Saving" button, and the program will start collecting data after the specified delay time and save the data to the `data/hand_dataset.csv` file. After the collection is completed, if the number of augmented data is set, the program will automatically perform data augmentation.

#### Double-gesture data collection
```bash
python collect2.py
```
The usage is similar to single-gesture data collection, and the data will be saved to the `data/hand2_dataset.csv` file.

### 2. Model Training
```bash
python train.py
```
This script will load the specified dataset (by default, it is `data/hand2_dataset.csv`), perform data preprocessing, model training, and evaluation, and select the best model to save in the `weight` directory. You can modify the `FILE_PATH` variable in `train.py` according to your needs to select different datasets.

### 3. Real-time Recognition
#### Single-gesture real-time recognition
```bash
python recognition1.py
```
The program will open the camera, recognize single gestures in real-time, and display the recognition results on the image.

#### Double-gesture real-time recognition
```bash
python recognition2.py
```
The program will open the camera, recognize double gestures in real-time, and display the recognition results on the image.

#### Real-time recognition of single and double gestures
```bash
python recognition.py
```
This script can handle both single-gesture and double-gesture recognition and automatically select the appropriate model for prediction based on the number of detected hands.

### 4. Data Visualization
```bash
python read_csv.py
```
This script can read the gesture data in the specified CSV file and visualize it. You can modify the `FILE_PATH` and `LINE_NUMBER` variables in `read_csv.py` to select different data for visualization.

## V. Notes
- Ensure that the camera is working properly, and perform data collection and recognition in a well-lit environment.
- The parameters for data augmentation (such as noise ratio, rotation angle range, etc.) can be adjusted in the `add_datas.py` file.
- The parameters for model training (such as model type, hyperparameter search range, etc.) can be modified in the `train.py` file.

## VI. Contribution
If you have any suggestions or improvements for this project, please feel free to submit a Pull Request or raise an Issue.
