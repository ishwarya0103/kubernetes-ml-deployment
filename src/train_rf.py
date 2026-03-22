import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("cifar10-random-forest")

    transform = transforms.ToTensor()

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

    train_size = 10000
    test_size = 2000

    X_train = []
    y_train = []

    for i in range(train_size):
        image, label = train_dataset[i]
        X_train.append(image.numpy().flatten())
        y_train.append(label)

    X_test = []
    y_test = []

    for i in range(test_size):
        image, label = test_dataset[i]
        X_test.append(image.numpy().flatten())
        y_test.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    with mlflow.start_run():
        mlflow.log_param("model_type", "random_forest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("train_size", train_size)
        mlflow.log_param("test_size", test_size)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_metric("test_accuracy", accuracy)

        print(f"Random Forest Accuracy: {accuracy:.4f}")

        os.makedirs("models", exist_ok=True)
        model_path = "models/rf_cifar10.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)


if __name__ == "__main__":
    main()