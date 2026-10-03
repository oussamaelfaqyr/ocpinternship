import os
import random
import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ==========================================================
# 0. REPRODUCIBILITY
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================================
# 1. PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "data processing", "data", "processed")
)

TRAIN_PATH = os.path.join(DATA_DIR, "train_final.npz")
VAL_PATH = os.path.join(DATA_DIR, "val_final.npz")
TEST_PATH = os.path.join(DATA_DIR, "test_final.npz")

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "gru_baseline.pt")

MLFLOW_DB = os.path.abspath(os.path.join(BASE_DIR, "mlflow.db"))
MLFLOW_URI = "sqlite:///" + MLFLOW_DB.replace("\\", "/")


# ==========================================================
# 2. HYPERPARAMETERS
# ==========================================================

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

PATIENCE = 7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# 3. HEADER
# ==========================================================

print("=" * 70)
print("OCP ELECTRICAL FAULT DETECTION - GRU BASELINE")
print("=" * 70)
print(f"Device: {DEVICE}\n")


# ==========================================================
# 4. LOAD DATA
# ==========================================================

print("=" * 70)
print("1. LOAD DATA")
print("=" * 70)

train = np.load(TRAIN_PATH)
val = np.load(VAL_PATH)
test = np.load(TEST_PATH)

X_train = train["X"].astype(np.float32)
y_train = train["y"].astype(np.int64)

X_val = val["X"].astype(np.float32)
y_val = val["y"].astype(np.int64)

X_test = test["X"].astype(np.float32)
y_test = test["y"].astype(np.int64)

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_val  : {X_val.shape}")
print(f"y_val  : {y_val.shape}")
print(f"X_test : {X_test.shape}")
print(f"y_test : {y_test.shape}\n")


# ==========================================================
# 5. DATASET CHECK
# ==========================================================

print("=" * 70)
print("2. DATASET CHECK")
print("=" * 70)

timesteps = X_train.shape[1]
n_features = X_train.shape[2]

print(f"Timesteps     : {timesteps}")
print(f"Features      : {n_features}")
print(f"Train samples : {len(X_train)}")
print(f"Val samples   : {len(X_val)}")
print(f"Test samples  : {len(X_test)}\n")

assert X_train.ndim == 3
assert X_val.ndim == 3
assert X_test.ndim == 3
assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
assert X_train.shape[2] == X_val.shape[2] == X_test.shape[2]


# ==========================================================
# 6. LABEL CHECK
# ==========================================================

print("=" * 70)
print("3. LABELS")
print("=" * 70)

classes = np.unique(y_train)
num_classes = len(classes)

print(f"Number of classes: {num_classes}\n")
print("Train labels:")
for c in classes:
    count = np.sum(y_train == c)
    print(f"  class {c}: {count}")
print()

unknown_val = sorted(set(np.unique(y_val)) - set(classes))
unknown_test = sorted(set(np.unique(y_test)) - set(classes))

if unknown_val:
    raise ValueError(f"Validation contains unseen labels: {unknown_val}")
if unknown_test:
    raise ValueError(f"Test contains unseen labels: {unknown_test}")


# ==========================================================
# 7. CLASS WEIGHTS
# ==========================================================

print("=" * 70)
print("4. CLASS WEIGHTS")
print("=" * 70)

beta = 0.999
class_counts = np.bincount(y_train, minlength=len(classes)).astype(float)
# Safeguard to prevent NaN/Inf if a class happens to be completely absent in train
class_counts = np.where(class_counts == 0, 1e-6, class_counts)

effective_num = 1.0 - np.power(beta, class_counts)
weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
weights = weights / np.sum(weights) * len(classes)

# Zero out weights for any class that isn't in the training set
weights = np.where(class_counts <= 1e-6, 0.0, weights)

class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

print("Class weights:")
for c, w in zip(classes, weights):
    print(f"  class {c}: {w:.4f}")
print()


# ==========================================================
# 8. PYTORCH DATASETS
# ==========================================================

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=torch.cuda.is_available()
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=torch.cuda.is_available()
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=torch.cuda.is_available()
)


# ==========================================================
# 9. MODEL
# ==========================================================

print("=" * 70)
print("5. MODEL")
print("=" * 70)

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, hidden = self.gru(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        return logits

model = GRUClassifier(
    input_size=n_features,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=num_classes,
    dropout=DROPOUT,
).to(DEVICE)

total_parameters = sum(p.numel() for p in model.parameters())
trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(model)
print()
print(f"Total parameters: {total_parameters:,}")
print(f"Trainable parameters: {trainable_parameters:,}\n")


# ==========================================================
# 10. LOSS + OPTIMIZER
# ==========================================================

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)


# ==========================================================
# 11. EVALUATION FUNCTION
# ==========================================================

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)

            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_targets = np.asarray(all_targets)
    all_predictions = np.asarray(all_predictions)

    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average="weighted", zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average="macro", zero_division=0
    )

    return avg_loss, accuracy, precision, recall, f1, macro_f1, all_targets, all_predictions


# ==========================================================
# 12. MLFLOW
# ==========================================================

print("=" * 70)
print("6. MLFLOW")
print("=" * 70)

print(f"MLflow database: {MLFLOW_DB}")
print(f"MLflow URI: {MLFLOW_URI}")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("PowerGrid_Baseline")
print("MLflow configured successfully.\n")


# ==========================================================
# 13. TRAINING
# ==========================================================

best_val_f1 = -np.inf
best_epoch = 0
epochs_without_improvement = 0
history = []

with mlflow.start_run(run_name="GRU_Baseline"):
    mlflow.log_params({
        "model": "GRU",
        "input_features": n_features,
        "timesteps": timesteps,
        "num_classes": num_classes,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW",
        "class_weighting": "effective_samples_beta_0.999",
        "seed": SEED,
    })

    print("=" * 70)
    print("7. TRAINING")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_macro_f1, _, _ = evaluate(model, val_loader)
        scheduler.step(val_macro_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "val_macro_f1": val_macro_f1,
        })

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1_weighted": val_f1,
            "val_f1_macro": val_macro_f1,
            "learning_rate": current_lr,
        }, step=epoch)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "input_size": n_features,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "num_classes": num_classes,
                "dropout": DROPOUT,
                "classes": classes.tolist(),
                "epoch": epoch,
                "val_macro_f1": val_macro_f1,
            }, MODEL_PATH)
            print(f"  -> Best model saved (epoch {epoch})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print()
    print("=" * 70)
    print("8. BEST MODEL")
    print("=" * 70)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation Macro-F1: {best_val_f1:.4f}\n")

    print("=" * 70)
    print("9. TEST")
    print("=" * 70)

    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_macro_f1, test_targets, test_predictions = evaluate(model, test_loader)

    print(f"Test Loss      : {test_loss:.4f}")
    print(f"Test Accuracy  : {test_accuracy:.4f}")
    print(f"Test Precision : {test_precision:.4f}")
    print(f"Test Recall    : {test_recall:.4f}")
    print(f"Test Weighted F1: {test_f1:.4f}")
    print(f"Test Macro F1  : {test_macro_f1:.4f}\n")

    print("=" * 70)
    print("10. CLASSIFICATION REPORT")
    print("=" * 70)
    report = classification_report(test_targets, test_predictions, labels=classes, zero_division=0)
    print(report)

    print("=" * 70)
    print("11. CONFUSION MATRIX")
    print("=" * 70)
    cm = confusion_matrix(test_targets, test_predictions, labels=classes)
    print(cm)
    print()

    print("=" * 70)
    print("12. PER-CLASS METRICS")
    print("=" * 70)
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        test_targets, test_predictions, labels=classes, zero_division=0
    )

    for i, c in enumerate(classes):
        print(
            f"Class {c:2d} | "
            f"Precision: {precision_per_class[i]:.4f} | "
            f"Recall: {recall_per_class[i]:.4f} | "
            f"F1: {f1_per_class[i]:.4f} | "
            f"Support: {support[i]}"
        )
    print()

    mlflow.log_metrics({
        "best_epoch": float(best_epoch),
        "best_val_macro_f1": best_val_f1,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1_weighted": test_f1,
        "test_f1_macro": test_macro_f1,
    })

    print("=" * 70)
    print("13. SAVE MODEL")
    print("=" * 70)

    print(f"PyTorch model saved to:\n{MODEL_PATH}")

    try:
        mlflow.pytorch.log_model(model, artifact_path="gru_model")
        print("GRU model logged to MLflow.")
    except Exception as e:
        print(f"WARNING: Could not log PyTorch model to MLflow: {e}")

    class_mapping = {int(c): int(c) for c in classes}
    joblib.dump(class_mapping, os.path.join(MODEL_DIR, "class_mapping.pkl"))

    np.save(os.path.join(MODEL_DIR, "confusion_matrix.npy"), cm)
    np.save(os.path.join(MODEL_DIR, "training_history.npy"), history, allow_pickle=True)

    mlflow.log_artifact(MODEL_PATH, artifact_path="saved_model")
    mlflow.log_artifact(os.path.join(MODEL_DIR, "confusion_matrix.npy"), artifact_path="evaluation")
    mlflow.log_artifact(os.path.join(MODEL_DIR, "training_history.npy"), artifact_path="training")
    print()

print("=" * 70)
print("14. FINAL RESULTS")
print("=" * 70)
print(f"Best validation Macro-F1 : {best_val_f1:.4f}")
print(f"Test Accuracy            : {test_accuracy:.4f}")
print(f"Test Weighted F1         : {test_f1:.4f}")
print(f"Test Macro-F1            : {test_macro_f1:.4f}\n")
print("Model:")
print(MODEL_PATH)
print("\nMLflow database:")
print(MLFLOW_DB)
print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)