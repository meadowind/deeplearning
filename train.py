import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

from dataset import get_dataloaders
from model import SimpleCNN, ResNet18Classifier, ViTClassifier, TransferMobileNet
from sklearn.metrics import f1_score

# ==========================================
# WINDOWS MULTIPROCESSING SHIELD
# ==========================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Quick initial data load just to dynamically count how many classes you have
    _, _, _, classes = get_dataloaders(batch_size=1, img_size=224, augmented=False)
    num_classes = len(classes)
    print(f"\n--- Detected {num_classes} classes ---")

    # ==========================================
    # 2. CHOOSE MODELS TO COMPARE
    # ==========================================
    # Add or comment out models in this list to include them in the final table!
    import os
    os.makedirs("baseline", exist_ok=True)

    models_to_compare = [
        SimpleCNN(num_classes=num_classes).to(device),          # NEW: baseline CNN from scratch
        ResNet18Classifier(num_classes=num_classes).to(device),  # NEW: pretrained ResNet-18
        TransferMobileNet(num_classes=num_classes).to(device),
        # ViTClassifier(num_classes=num_classes).to(device),     # heavy, needs lots of RAM
    ]

    num_epochs = 15 
    phase_2_start_epoch = 5 

    summary_results = []

    print("\n" + "="*60)
    print(f"{'STARTING BASELINE MODEL SELECTION':^60}")
    print("="*60)

    # ==========================================
    # 3. AUTOMATED EVALUATION LOOP
    # ==========================================
    for model in models_to_compare:
        model_name = model.__class__.__name__
        
        # Auto-set image size: 64 for SimpleCNN (trained from scratch), 224 for pretrained models
        image_size = 64 if isinstance(model, SimpleCNN) else 224
        print(f"\n>>> Training {model_name} (Input: {image_size}x{image_size})")

        train_loader, val_loader, test_loader, class_names = get_dataloaders(
            batch_size=32, 
            img_size=image_size, 
            augmented=True
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

        best_val_accuracy = 0.0
        start_time = time.time() 

        # Lists to store history for graphing
        history_train_loss, history_val_loss = [], []
        history_train_acc, history_val_acc = [], []

        for epoch in range(num_epochs):
            print(f"  Epoch {epoch+1}/{num_epochs}...", end=" ")
            
            # --- Phase 2 Unfreezing for TransferMobileNet ---
            if isinstance(model, TransferMobileNet) and epoch == phase_2_start_epoch:
                model.unfreeze_last_n_blocks(n=2)
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

            # --- Phase 2 Unfreezing for ResNet18Classifier ---
            # Unfreeze the whole backbone at phase_2_start_epoch with a lower LR
            if isinstance(model, ResNet18Classifier) and epoch == phase_2_start_epoch:
                for param in model.backbone.parameters():
                    param.requires_grad = True
                optimizer = optim.Adam(model.parameters(), lr=0.0001)
                print(f"  [ResNet18] Unfroze full backbone at epoch {epoch+1}, LR → 0.0001")

            # --- Training Step ---
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_loss_epoch = running_loss / len(train_loader)
            train_acc_epoch = 100 * correct_train / total_train

            # --- Validation Step ---
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            # Store predictions for the confusion matrix
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
            val_loss_epoch = val_loss / len(val_loader)
            val_acc_epoch = 100 * correct_val / total_val
            
            print(f"Val Acc: {val_acc_epoch:.2f}%")
            
            # Save epoch data to our history lists
            history_train_loss.append(train_loss_epoch)
            history_val_loss.append(val_loss_epoch)
            history_train_acc.append(train_acc_epoch)
            history_val_acc.append(val_acc_epoch)
            
            if val_acc_epoch > best_val_accuracy:
                best_val_accuracy = val_acc_epoch
                torch.save(model.state_dict(), f"baseline/best_{model_name}.pth")

        total_training_time = time.time() - start_time

        # Compute macro F1 from the last epoch's val predictions
        best_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100

        summary_results.append({
            "Model": model_name,
            "Accuracy": best_val_accuracy,
            "F1": best_f1,
            "Time": total_training_time
        })

        # ==========================================
        # GRAPH GENERATION
        # ==========================================
        print(f"  Generating graphs for {model_name}...")
        
        # 1. Plot Training vs Validation Loss & Accuracy
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(history_train_loss, label='Train Loss', color='blue')
        ax1.plot(history_val_loss, label='Val Loss', color='red', linestyle='--')
        ax1.set_title(f'{model_name}: Loss over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ax2.plot(history_train_acc, label='Train Acc', color='blue')
        ax2.plot(history_val_acc, label='Val Acc', color='red', linestyle='--')
        ax2.set_title(f'{model_name}: Accuracy over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.savefig(f"baseline/{model_name}_metrics.png", bbox_inches='tight')
        plt.close() 

        # 2. Plot Confusion Matrix
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(all_labels, all_preds)
        
        sns.heatmap(cm, annot=False, cmap='Blues', fmt='d', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name}: Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        
        plt.savefig(f"baseline/{model_name}_confusion_matrix.png", bbox_inches='tight')
        plt.close()

    # ==========================================
    # 4. PRINT THE FINAL SUMMARY TABLE
    # ==========================================
    print("\n\n" + "="*70)
    print(f"{'--- Baseline Model Selection Results ---':^70}")
    print("="*70)
    print(f"{'Model':<25} | {'Best Accuracy (%)':<20} | {'Macro F1 (%)':<14} | {'Total Time (s)':<10}")
    print("-" * 70)
    
    for result in summary_results:
        print(f"{result['Model']:<25} | {result['Accuracy']:<20.2f} | {result['F1']:<14.2f} | {result['Time']:<10.2f}")
    
    print("="*70 + "\n")