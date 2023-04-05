import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from joblib import Parallel, delayed
from functools import partial
from tqdm import tqdm

def load_image(data_path, path):
    img = cv2.imread(os.path.join(data_path, path), cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:  # Check if the image has color channels
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, (256, 256))

def load_images_and_labels(data_path, csv_path, n_threads=4):
    df = pd.read_csv(csv_path)
    df['label'] = df['Finding Labels'].apply(lambda x: 1 if x != 'No Finding' else 0)
    image_paths = df['Image Index'][0]
    labels = df['label'].values
    # 读取图像
    img = cv2.imread(os.path.join(data_path, image_paths))

    # 检查图像的通道数
    channels = img.shape[2]


    print("图像的通道数为:", channels)

def preprocess_images(images):
    return images.reshape(images.shape[0], -1)

def main():
    data_path = ''
    csv_path = 'data_1.csv'
    n_threads = 4

    print("Loading images and labels...")
    images, labels = load_images_and_labels(data_path, csv_path, n_threads)
    images = preprocess_images(images)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)



if __name__ == '__main__':
    main()
