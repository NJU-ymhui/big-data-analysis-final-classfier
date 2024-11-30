import torch
import numpy as np


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device="cpu", epochs=20):
    print("Training on", device)
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            # 前向传播
            features = features.float()  # 转换为float, 同一类型
            labels = labels.long()  # 转换为long, cuda supported
            outputs = model(features)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        train_acc = correct / total

        # 验证
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                # 类型转换
                features = features.float()
                labels = labels.long()
                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch {epoch + 1}/{epochs}: "
              f"Train Loss={train_loss / len(train_loader): .4f}, Train Acc={train_acc: .4f}, "
              f"Val Loss={val_loss / len(val_loader): .4f}, Val Acc={val_acc: .4f}")


def predict(model, test_loader, device="cpu"):
    print("Predicting on", device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in test_loader:
            # 类型转换
            features = features.float()
            features = features.to(device)
            outputs = model(features)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
    return np.array(predictions)  # TODO try torch.array(predictions)
