import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import mlflow


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("cifar10-cnn")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5

    with mlflow.start_run():
        mlflow.log_param("model_type", "cnn")
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("num_epochs", num_epochs)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)

            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch + 1)
            mlflow.log_metric("test_accuracy", accuracy, step=epoch + 1)

            print(
                f"Epoch [{epoch+1}/{num_epochs}] - "
                f"Train Loss: {avg_train_loss:.4f} - "
                f"Test Accuracy: {accuracy:.2f}%"
            )

        os.makedirs("models", exist_ok=True)
        model_path = "models/cnn_cifar10.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)


if __name__ == "__main__":
    main()