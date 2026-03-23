# train.py
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time

#  CNN Model 
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 → 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 28x28 → 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 28x28 → 14x14
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# 14x14 → 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 14x14 → 7x7
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.conv_block(x))


#  Config 
EPOCHS      = 5
BATCH_SIZE  = 64
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR    = "./data"

#  Data 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))   # MNIST mean/std
])

train_dataset = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

#  Helpers 
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += outputs.argmax(1).eq(labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct    += outputs.argmax(1).eq(labels).sum().item()
            total      += images.size(0)
    return total_loss / total, correct / total


#  Main 
def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("mnist-cnn")

    model     = MNISTNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    print(f"Training on: {DEVICE}")
    print(f"Train samples: {len(train_dataset):,} | Test samples: {len(test_dataset):,}\n")

    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_params({
            "epochs":     EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr":         LR,
            "optimizer":  "Adam",
            "scheduler":  "StepLR(step=2, gamma=0.5)",
            "device":     str(DEVICE),
            "model":      "MNISTNet-CNN"
        })

        best_acc = 0.0

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
            val_loss,   val_acc   = evaluate(model, test_loader, criterion)
            scheduler.step()
            elapsed = time.time() - t0

            # Log per-epoch metrics
            mlflow.log_metrics({
                "train_loss": round(train_loss, 4),
                "train_acc":  round(train_acc,  4),
                "val_loss":   round(val_loss,   4),
                "val_acc":    round(val_acc,    4),
            }, step=epoch)

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
                f"⏱ {elapsed:.1f}s"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_model.pt")

        # Log final accuracy (used by check_threshold.py)
        mlflow.log_metric("accuracy", best_acc)

        # Log the best model artifact
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="mnist-cnn"
        )
        mlflow.log_artifact("best_model.pt")

        print(f"\n Best Val Accuracy: {best_acc:.4f}")
        print(f"   Run ID: {run.info.run_id}")

        # Write Run ID for the deploy job
        with open("model_info.txt", "w") as f:
            f.write(run.info.run_id)


if __name__ == "__main__":
    main()