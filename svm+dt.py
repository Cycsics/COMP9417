import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from joblib import Parallel, delayed
from functools import partial
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import trange

def load_image(data_path, path):
    img = cv2.imread(os.path.join(data_path, path), cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:  # Check if the image has color channels
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, (256, 256))

def load_images_and_labels(data_path, csv_path, data_fraction=1.0, n_threads=4):
    df = pd.read_csv(csv_path)
    df['label'] = df['Finding Labels'].apply(lambda x: 1 if x != 'No Finding' else 0)

    if data_fraction < 1.0:
        num_samples = int(len(df) * data_fraction)
        df = df.sample(num_samples, random_state=42)

    image_paths = df['Image Index'].values
    labels = df['label'].values

    images = Parallel(n_jobs=n_threads)(delayed(load_image)(data_path, path) for path in tqdm(image_paths, desc="Loading images"))
    return np.array(images), labels


def preprocess_images(images):
    return images.reshape(images.shape[0], -1)

def main(data_fraction=1.0):
    assert 0 < data_fraction <= 1, "Data fraction should be between 0 and 1."

    data_path = ''
    csv_path = 'data_1.csv'
    n_threads = 24

    print("Loading images and labels...")
    images, labels = load_images_and_labels(data_path, csv_path, data_fraction, n_threads)
    
    if data_fraction < 1.0:
        num_samples = int(len(images) * data_fraction)
        indices = np.random.choice(len(images), num_samples, replace=False)
        images = images[indices]
        labels = labels[indices]

    images = preprocess_images(images)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training SVM and Decision Tree...")
    svm_clf = SVC(verbose=True)
    tree_clf = DecisionTreeClassifier()

    # Train SVM and Decision Tree separately and display progress
    print("Training SVM...")
    with logging_redirect_tqdm():
        for _ in trange(10):  # 迭代次数为10，可以根据具体情况修改
            svm_clf.fit(X_train, y_train)

    print("Training Decision Tree...")
    with logging_redirect_tqdm():
        for _ in trange(10):  # 迭代次数为10，可以根据具体情况修改
            tree_clf.fit(X_train, y_train)

    # Combine trained classifiers using VotingClassifier
    combined_clf = VotingClassifier(estimators=[('svm', svm_clf), ('tree', tree_clf)], voting='hard')
    combined_clf.fit(X_train, y_train)  # No need to train again, as base classifiers are already trained

    y_pred_combined = combined_clf.predict(X_test)
    print("\nCombined Classifier (SVM and Decision Tree):")
    print(classification_report(y_test, y_pred_combined))

if __name__ == '__main__':
    main(data_fraction=0.5)  # Use 50% of the dataset