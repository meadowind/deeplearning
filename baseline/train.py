from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)

from dataset import get_dataloaders
from model import SimpleCNN, ResNet18Classifier, MobileNetV2Classifier
# Config
base_dir = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "MobileNetV2"      # "SimpleCNN" / "ResNet18/ MobileNetV2"
augmentation = True     # True / False

num_epoch = 15
learning_rate = 0.001
batch_size = 32
img_size = 64
early_stopping_patience = 5
pretrained = True

data_root = base_dir / "cropped_belgiumts_classid"
output_dir = base_dir / f"outputs_{model_name}_aug{augmentation}"
output_dir.mkdir(exist_ok=True)

def build_model(model_name: str, num_classes: int):
    if model_name == "SimpleCNN":
        return SimpleCNN(num_classes=num_classes)
    elif model_name == "ResNet18":
        return ResNet18Classifier(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'mobilenet':
        return  MobileNetV2Classifier(num_classes=47, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported MODEL_NAME: {model_name}")


def evaluate_model(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, macro_f1, all_labels, all_preds


def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_metric_curve(val_accs, val_f1s, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(val_accs, label="Val Accuracy")
    plt.plot(val_f1s, label="Val Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train():
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        img_size=img_size,
        val_ratio=0.2,
        augmented=augmentation,
        num_workers=2,
    )

    num_classes = len(class_names)

    print("=" * 50)
    print(f"Model Name: {model_name}")
    print(f"Use Augmentation: {augmentation}")
    print(f"Num Classes: {num_classes}")
    print(f"Output Dir: {output_dir}")
    print("=" * 50)

    model = build_model(model_name, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []

    best_val_f1 = -1.0
    patience_counter = 0
    best_model_path = output_dir / f"best_{model_name}.pth"

    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_acc, val_f1, _, _ = evaluate_model(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(
            f"Epoch [{epoch+1}/{num_epoch}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    plot_loss_curve(train_losses, val_losses, output_dir / "loss_curve.png")
    plot_metric_curve(val_accs, val_f1s, output_dir / "val_metrics.png")

    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate_model(model, test_loader, criterion)

    print("\n FINAL TEST RESULTS:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    print("\nClassification Report:\n")
    print(report)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_path=output_dir / "confusion_matrix.png",
    )

    with open(output_dir / "final_test_results.txt", "w", encoding="utf-8") as f:
        f.write("FINAL TEST RESULTS: \n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Augmentation: {augmentation}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Macro F1: {test_f1:.4f}\n")

    with open(output_dir / "run_config.txt", "w", encoding="utf-8") as f:
        f.write(f"MODEL_NAME={model_name}\n")
        f.write(f"USE_AUGMENTATION={augmentation}\n")
        f.write(f"NUM_EPOCHS={num_epoch}\n")
        f.write(f"LEARNING_RATE={learning_rate}\n")
        f.write(f"BATCH_SIZE={batch_size}\n")
        f.write(f"IMG_SIZE={img_size}\n")
        f.write(f"PRETRAINED={pretrained}\n")

    print(f"\nAll outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    train()