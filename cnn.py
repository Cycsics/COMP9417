import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


# Custom dataset class
class NIH_Dataset(Dataset):
    def __init__(self, labels_frame, img_dir, transform=None):
        self.labels_frame = labels_frame
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.labels_frame.iloc[idx, -1]

        if self.transform:
            image = self.transform(image)

        return image, label
    
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # 自动生成不重复的文件名
    file_index = 0
    while os.path.exists(f"performance_chart_{file_index}.png"):
        file_index += 1
    plt.savefig(f"performance_chart_{file_index}.png")
    plt.show()

def main():
    # Load and split the data
    total_start_time = time.time()
    csv_path = 'data_1.csv'
    data_frame = pd.read_csv(csv_path)
    data_frame['label'] = data_frame['Finding Labels'].apply(lambda x: 1 if x != 'No Finding' else 0)
    train_frame, test_frame = train_test_split(data_frame, test_size=0.2, random_state=42)
    train_frame, val_frame = train_test_split(train_frame, test_size=0.25, random_state=42)

    # Transforms
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    batch_size = 64
    # Create datasets and data loaders
    img_dir = ''
    train_dataset = NIH_Dataset(train_frame, img_dir, transform=data_transforms)
    val_dataset = NIH_Dataset(val_frame, img_dir, transform=data_transforms)
    test_dataset = NIH_Dataset(test_frame, img_dir, transform=data_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Rest of the code for model training remains the same
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load a pre-trained ResNet-50 model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Replace the last layer with a new fully connected layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

    # Move the model to the appropriate device
    model = model.to(device)

    # Define the loss function and the optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 1

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        epoch_end_time = time.time()  # 记录epoch结束时间
        epoch_duration = end_time - start_time  # 计算epoch所需的时间
        print(f"Epoch {epoch+1} took {epoch_duration:.2f} seconds.")
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        model.eval()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = running_corrects.double() / len(val_dataset)

        val_losses.append(epoch_loss)
        val_accuracies.append(epoch_acc.item())

    print("Training complete.")
    model.eval()  # 将模型设置为评估模式
    y_true = []
    y_pred = []

    with torch.no_grad():  # 禁用梯度计算以节省计算资源
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入和标签发送到设备（例如GPU）
            outputs = model(inputs)  # 计算输出
            _, predictions = torch.max(outputs, 1)  # 获得预测结果

            y_true.extend(labels.cpu().numpy())  # 将真实标签添加到y_true列表中
            y_pred.extend(predictions.cpu().numpy())  # 将预测结果添加到y_pred列表中

    # 使用sklearn的classification_report方法计算性能指标
    print("Model performance:")
    print(classification_report(y_true, y_pred))
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
    total_end_time = time.time()
    print(f"Total time taken: {total_end_time - total_start_time:.2f} seconds.")
    
    
if __name__ == '__main__':
    main()

