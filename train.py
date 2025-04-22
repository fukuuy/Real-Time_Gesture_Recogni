import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
import joblib


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values  # 所有特征列
    y = df.iloc[:, -1].values  # 标签列
    return X, y


def preprocess_data(X, y, test_size=0.2, random_state=42):
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, model_type='random_forest'):
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
    elif model_type == 'svm':
        model = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            class_weight='balanced'
        )
    elif model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError("未知的模型类型")

    model.fit(X_train, y_train)
    return model


# 4. 评估模型
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"模型准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(report)

    return accuracy


if __name__ == "__main__":
    csv_path = "hand_dataset.csv"
    X, y = load_data(csv_path)  #


    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    models = {
        'random_forest': "随机森林",
        'svm': "支持向量机",
        'knn': "K近邻"
    }

    best_model = None
    best_accuracy = 0

    for model_type, name in models.items():
        print(f"\n训练 {name} 模型")
        model = train_model(X_train, y_train, model_type)
        accuracy = evaluate_model(model, X_test, y_test)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    # 保存最佳模型
    if best_model is not None:
        joblib.dump(best_model, 'gesture_model.pkl')
        print("\n最佳模型已保存为 'gesture_model.pkl'")
