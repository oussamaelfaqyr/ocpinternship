"""
======================================================================
E4-B: Temporal Attention Pooling  |  Focal Loss (gamma=2.0)  |  WRS
======================================================================
Architecture:
    - GRU(input_size=35, hidden_size=128, num_layers=2, dropout=0.2)
    - Temporal Attention Pooling (soft-attention over all 20 timesteps)
    - Sampler: WeightedRandomSampler(alpha=0.5)
    - Loss: Multi-class Focal Loss (gamma=2.0)
Goal:
    Dynamically focus on the exact timestamp where the fault transient occurs
    (e.g., short-circuit spike, load step) anywhere in the 20s window.
======================================================================
"""

import os
import random
import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
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
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.abspath(os.path.join(BASE_DIR, "..", "data processing", "data", "processed"))
TRAIN_PATH = os.path.join(DATA_DIR, "train_final.npz")
VAL_PATH   = os.path.join(DATA_DIR, "val_final.npz")
TEST_PATH  = os.path.join(DATA_DIR, "test_final.npz")
MODEL_DIR  = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "gru_e4b.pt")
MLFLOW_DB  = os.path.abspath(os.path.join(BASE_DIR, "mlflow.db"))
MLFLOW_URI = "sqlite:///" + MLFLOW_DB.replace("\\", "/")

# ==========================================================
# 2. HYPERPARAMETERS
# ==========================================================
BATCH_SIZE    = 128
EPOCHS        = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-4

HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
DROPOUT       = 0.2
PATIENCE      = 7
WRS_ALPHA     = 0.5
FOCAL_GAMMA   = 2.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print(f"E4-B: GRU hidden=128  |  Temporal Attention Pooling  |  Focal Loss (gamma={FOCAL_GAMMA})")
print("=" * 70)
print(f"Device: {DEVICE}\n")

# ==========================================================
# 3. LOAD DATA
# ==========================================================
print("=" * 70); print("1. LOAD DATA"); print("=" * 70)
train = np.load(TRAIN_PATH)
val   = np.load(VAL_PATH)
test  = np.load(TEST_PATH)

X_train = train["X"].astype(np.float32); y_train = train["y"].astype(np.int64)
X_val   = val["X"].astype(np.float32);   y_val   = val["y"].astype(np.int64)
X_test  = test["X"].astype(np.float32);  y_test  = test["y"].astype(np.int64)

print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"X_val  : {X_val.shape}   | y_val  : {y_val.shape}")
print(f"X_test : {X_test.shape}  | y_test : {y_test.shape}\n")

timesteps   = X_train.shape[1]
n_features  = X_train.shape[2]
classes     = np.unique(y_train)
num_classes = len(classes)

# ==========================================================
# 4. WEIGHTED RANDOM SAMPLER
# ==========================================================
class_counts = np.bincount(y_train, minlength=num_classes).astype(float)
counts_safe   = np.maximum(class_counts, 1)
class_wrs_w   = 1.0 / (counts_safe ** WRS_ALPHA)
class_wrs_w  /= class_wrs_w.sum()

sample_weights = torch.tensor(class_wrs_w[y_train], dtype=torch.float64)
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
    generator=torch.Generator().manual_seed(SEED),
)

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
test_dataset  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=torch.cuda.is_available())
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=torch.cuda.is_available())
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=torch.cuda.is_available())

# ==========================================================
# 5. FOCAL LOSS DEFINITION
# ==========================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        ce_loss = -log_pt
        loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = loss * alpha_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# ==========================================================
# 6. MODEL WITH TEMPORAL ATTENTION POOLING
# ==========================================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False)
        )

    def forward(self, gru_seq):
        # gru_seq: (batch, timesteps, hidden_size)
        scores = self.net(gru_seq)                    # (batch, timesteps, 1)
        weights = F.softmax(scores, dim=1)            # (batch, timesteps, 1)
        context = torch.sum(gru_seq * weights, dim=1) # (batch, hidden_size)
        return context, weights

class AttentiveGRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        gru_seq, _ = self.gru(x)               # (batch, 20, hidden_size)
        context, _ = self.attention(gru_seq)   # (batch, hidden_size)
        context = self.dropout(context)
        return self.fc(context)

model = AttentiveGRUClassifier(
    input_size=n_features, hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS, num_classes=num_classes, dropout=DROPOUT,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}\n")

criterion = FocalLoss(gamma=FOCAL_GAMMA).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

# ==========================================================
# 7. EVALUATION FUNCTION
# ==========================================================
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits = model(Xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * Xb.size(0)
            all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    t, p = np.asarray(all_targets), np.asarray(all_preds)
    acc  = accuracy_score(t, p)
    prec, rec, f1, _ = precision_recall_fscore_support(t, p, average="weighted", zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(t, p, average="macro", zero_division=0)
    return avg_loss, acc, prec, rec, f1, macro_f1, t, p

# ==========================================================
# 8. MLFLOW + TRAINING
# ==========================================================
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("PowerGrid_E4")

best_val_f1 = -np.inf
best_epoch  = 0
no_improve  = 0

with mlflow.start_run(run_name="E4B_AttentiveGRU_FocalLoss_gamma2.0_WRS"):
    mlflow.log_params({
        "experiment": "E4-B",
        "model": "AttentiveGRU",
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "sampler": f"WeightedRandomSampler_alpha{WRS_ALPHA}",
        "loss": f"FocalLoss_gamma{FOCAL_GAMMA}",
        "input_features": n_features,
        "pooling": "temporal_attention",
    })

    print("=" * 70); print("5. TRAINING"); print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_macro_f1, _, _ = evaluate(model, val_loader)
        scheduler.step(val_macro_f1)
        lr_now = optimizer.param_groups[0]["lr"]

        mlflow.log_metrics({
            "train_loss": train_loss, "val_loss": val_loss,
            "val_acc": val_acc, "val_f1_weighted": val_f1,
            "val_f1_macro": val_macro_f1, "lr": lr_now,
        }, step=epoch)

        print(f"Epoch {epoch:02d}/{EPOCHS}  train={train_loss:.4f}  "
              f"val={val_loss:.4f}  acc={val_acc:.4f}  "
              f"wF1={val_f1:.4f}  macroF1={val_macro_f1:.4f}  lr={lr_now:.6f}")

        if val_macro_f1 > best_val_f1:
            best_val_f1, best_epoch, no_improve = val_macro_f1, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_size": n_features, "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS, "num_classes": num_classes,
                "dropout": DROPOUT, "classes": classes.tolist(),
                "epoch": epoch, "val_macro_f1": val_macro_f1,
            }, MODEL_PATH)
            print(f"  -> Best saved (epoch {epoch}, val_macro_f1={val_macro_f1:.4f})")
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stop at epoch {epoch}.")
            break

    # ==========================================================
    # 9. TEST (once on best checkpoint)
    # ==========================================================
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\n" + "=" * 70); print("6. TEST RESULTS"); print("=" * 70)
    test_loss, test_acc, test_prec, test_rec, test_f1, test_macro_f1, t_targets, t_preds = evaluate(model, test_loader)

    print(f"Best epoch      : {best_epoch}")
    print(f"Best val MacroF1: {best_val_f1:.4f}")
    print(f"Test Accuracy   : {test_acc:.4f}")
    print(f"Test Weighted F1: {test_f1:.4f}")
    print(f"Test Macro F1   : {test_macro_f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(t_targets, t_preds, labels=classes, zero_division=0))

    cm = confusion_matrix(t_targets, t_preds, labels=classes)
    print("Confusion Matrix:")
    print(cm)

    print("\nPer-class detail:")
    p_cls, r_cls, f1_cls, sup = precision_recall_fscore_support(
        t_targets, t_preds, labels=classes, zero_division=0)
    label_names = {0:"normal",1:"lg_fault",2:"ll_fault",3:"over_freq",
                   4:"overload",5:"transf_trip",6:"under_freq",7:"voltage_sag"}
    for i, c in enumerate(classes):
        print(f"  class {c} {label_names.get(c,'?'):15} | "
              f"P={p_cls[i]:.3f} R={r_cls[i]:.3f} F1={f1_cls[i]:.3f} sup={sup[i]}")

    mlflow.log_metrics({
        "best_val_macro_f1": best_val_f1,
        "test_acc": test_acc, "test_f1_weighted": test_f1, "test_macro_f1": test_macro_f1,
    })
    np.save(os.path.join(MODEL_DIR, "confusion_matrix_e4b.npy"), cm)
    mlflow.log_artifact(os.path.join(MODEL_DIR, "confusion_matrix_e4b.npy"))

print("\n" + "=" * 70)
print("E4-B COMPLETE")
print("=" * 70)
