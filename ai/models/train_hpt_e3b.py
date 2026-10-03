"""
======================================================================
OCP ELECTRICAL FAULT DETECTION - OPTUNA HPT BASED ON CHAMPION E3-B
======================================================================
Architecture:
    Input (35 features)
         ↓
    GRU(hidden_size, num_layers, dropout)
         ↓
    last hidden state (output[:, -1, :])
         ↓
    Dropout(dropout)
         ↓
    Linear(hidden_size, 8)

Training protocol:
    - WeightedRandomSampler(alpha in [0.3, 0.7])
    - CrossEntropyLoss() (unweighted)
    - Optimizer: AdamW + ReduceLROnPlateau
    - Objective: Maximiser val_macro_f1 (strict split: TRAIN -> VAL)
    - Test evaluation: ONE-SHOT at the very end on the retrained best model.
======================================================================
"""

import os
import random
import joblib
import optuna
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# Suppress verbose Optuna logs (keep only Trial summaries)
optuna.logging.set_verbosity(optuna.logging.INFO)

# ==========================================================
# 0. REPRODUCIBILITY & DEVICE
# ==========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# 1. PATHS & MLFLOW
# ==========================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.abspath(os.path.join(BASE_DIR, "..", "data processing", "data", "processed"))
TRAIN_PATH = os.path.join(DATA_DIR, "train_final.npz")
VAL_PATH   = os.path.join(DATA_DIR, "val_final.npz")
TEST_PATH  = os.path.join(DATA_DIR, "test_final.npz")

MODEL_DIR  = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "gru_optuna_e3b_best.pt")

MLFLOW_DB  = os.path.abspath(os.path.join(BASE_DIR, "mlflow.db"))
MLFLOW_URI = "sqlite:///" + MLFLOW_DB.replace("\\", "/")

mlflow.set_tracking_uri(MLFLOW_URI)
EXPERIMENT_NAME = "PowerGrid_Optuna_E3B"
mlflow.set_experiment(EXPERIMENT_NAME)

# ==========================================================
# 2. LOAD DATA (TRAIN, VAL, TEST)
# ==========================================================
print("=" * 70)
print("OCP FAULT DETECTION - OPTUNA HPT (E3-B BASELINE)")
print("=" * 70)
print(f"Device: {DEVICE}")

train = np.load(TRAIN_PATH)
val   = np.load(VAL_PATH)
test  = np.load(TEST_PATH)

X_train = train["X"].astype(np.float32); y_train = train["y"].astype(np.int64)
X_val   = val["X"].astype(np.float32);   y_val   = val["y"].astype(np.int64)
X_test  = test["X"].astype(np.float32);  y_test  = test["y"].astype(np.int64)

timesteps   = X_train.shape[1]
n_features  = X_train.shape[2]
classes     = np.unique(y_train)
num_classes = len(classes)

print(f"Input Features : {n_features} (E3-B Clean)")
print(f"Timesteps      : {timesteps}")
print(f"Train samples  : {len(X_train)}")
print(f"Val samples    : {len(X_val)}")
print(f"Test samples   : {len(X_test)}")
print(f"Classes        : {num_classes}")
print("=" * 70)

# Pre-calculate class counts for WRS
class_counts = np.bincount(y_train, minlength=num_classes).astype(float)
counts_safe  = np.maximum(class_counts, 1)

# PyTorch Tensor Datasets
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
test_dataset  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

# ==========================================================
# 3. MODEL ARCHITECTURE
# ==========================================================
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
        output, _ = self.gru(x)
        last = self.dropout(output[:, -1, :])
        return self.fc(last)

# ==========================================================
# 4. EVALUATION HELPER
# ==========================================================
def evaluate(model, loader, criterion):
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
# 5. OPTUNA OBJECTIVE FUNCTION
# ==========================================================
def objective(trial):
    # Hyperparameter search space
    hidden_size   = trial.suggest_categorical("hidden_size", [128, 160, 192, 256])
    num_layers    = trial.suggest_int("num_layers", 1, 3)
    dropout       = trial.suggest_float("dropout", 0.05, 0.35, step=0.05)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 8e-4, log=True)
    weight_decay  = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    wrs_alpha     = trial.suggest_float("wrs_alpha", 0.30, 0.70, step=0.05)
    batch_size    = trial.suggest_categorical("batch_size", [64, 128, 256])
    
    # Build WRS sampler for this trial's alpha
    class_wrs_w = 1.0 / (counts_safe ** wrs_alpha)
    class_wrs_w /= class_wrs_w.sum()
    sample_weights = torch.tensor(class_wrs_w[y_train], dtype=torch.float64)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    
    # Model
    model = GRUClassifier(
        input_size=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    
    TRIAL_EPOCHS = 20
    best_trial_val_f1 = -np.inf
    epochs_no_improve = 0
    PATIENCE = 5
    
    with mlflow.start_run(run_name=f"Trial_{trial.number:02d}", nested=True):
        mlflow.log_params(trial.params)
        
        for epoch in range(1, TRIAL_EPOCHS + 1):
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
            val_loss, val_acc, val_prec, val_rec, val_f1, val_macro_f1, _, _ = evaluate(model, val_loader, criterion)
            scheduler.step(val_macro_f1)
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_macro_f1,
            }, step=epoch)
            
            if val_macro_f1 > best_trial_val_f1:
                best_trial_val_f1 = val_macro_f1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            # Optuna pruning report
            trial.report(val_macro_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
            if epochs_no_improve >= PATIENCE:
                break
                
        mlflow.log_metric("best_val_macro_f1", best_trial_val_f1)
        
    return best_trial_val_f1

# ==========================================================
# 6. RUN OPTUNA STUDY
# ==========================================================
def main():
    N_TRIALS = 30
    
    print("\n" + "=" * 70)
    print(f"STARTING OPTUNA STUDY ({N_TRIALS} TRIALS)")
    print("=" * 70)
    
    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=6)
    
    study = optuna.create_study(
        study_name="GRU_E3B_Optimization",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    
    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS, timeout=None)
    
    print("\n" + "=" * 70)
    print("OPTUNA STUDY COMPLETED")
    print("=" * 70)
    print(f"Number of finished trials : {len(study.trials)}")
    print(f"Best trial number         : #{study.best_trial.number}")
    print(f"Best Validation Macro-F1  : {study.best_value:.4f}")
    print("\nBest Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k:15}: {v}")
    print("=" * 70)
    
    # ==========================================================
    # 7. FINAL RETRAIN ON BEST HYPERPARAMETERS (30 EPOCHS)
    # ==========================================================
    print("\n" + "=" * 70)
    print("RETRAINING FINAL MODEL ON BEST HYPERPARAMETERS (30 EPOCHS)")
    print("=" * 70)
    
    best_p = study.best_params
    best_hidden     = best_p["hidden_size"]
    best_layers     = best_p["num_layers"]
    best_dropout    = best_p["dropout"]
    best_lr         = best_p["learning_rate"]
    best_wd         = best_p["weight_decay"]
    best_alpha      = best_p["wrs_alpha"]
    best_batch_size = best_p["batch_size"]
    
    # Sampler for best alpha
    class_wrs_w = 1.0 / (counts_safe ** best_alpha)
    class_wrs_w /= class_wrs_w.sum()
    sample_weights = torch.tensor(class_wrs_w[y_train], dtype=torch.float64)
    final_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    
    final_train_loader = DataLoader(
        train_dataset, batch_size=best_batch_size, sampler=final_sampler,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    final_val_loader = DataLoader(
        val_dataset, batch_size=best_batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    final_test_loader = DataLoader(
        test_dataset, batch_size=best_batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    
    final_model = GRUClassifier(
        input_size=n_features,
        hidden_size=best_hidden,
        num_layers=best_layers,
        num_classes=num_classes,
        dropout=best_dropout,
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_lr, weight_decay=best_wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    
    FINAL_EPOCHS = 30
    best_final_val_f1 = -np.inf
    best_final_epoch = 0
    final_no_improve = 0
    PATIENCE = 7
    
    with mlflow.start_run(run_name="Final_Best_Optuna_Model"):
        mlflow.log_params(best_p)
        mlflow.log_param("architecture", "GRU_E3B_Optimized")
        
        for epoch in range(1, FINAL_EPOCHS + 1):
            final_model.train()
            running_loss = 0.0
            for Xb, yb in final_train_loader:
                Xb, yb = Xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()
                loss = criterion(final_model(Xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
                optimizer.step()
                running_loss += loss.item() * Xb.size(0)
                
            train_loss = running_loss / len(final_train_loader.dataset)
            val_loss, val_acc, val_prec, val_rec, val_f1, val_macro_f1, _, _ = evaluate(final_model, final_val_loader, criterion)
            scheduler.step(val_macro_f1)
            lr_now = optimizer.param_groups[0]["lr"]
            
            print(f"Final Epoch {epoch:02d}/{FINAL_EPOCHS} | Train Loss: {train_loss:.4f} | Val Macro-F1: {val_macro_f1:.4f} | LR: {lr_now:.6f}")
            
            if val_macro_f1 > best_final_val_f1:
                best_final_val_f1 = val_macro_f1
                best_final_epoch = epoch
                final_no_improve = 0
                torch.save({
                    "model_state_dict": final_model.state_dict(),
                    "input_size": n_features,
                    "hidden_size": best_hidden,
                    "num_layers": best_layers,
                    "num_classes": num_classes,
                    "dropout": best_dropout,
                    "classes": classes.tolist(),
                    "epoch": epoch,
                    "val_macro_f1": val_macro_f1,
                    "best_params": best_p,
                }, BEST_MODEL_PATH)
            else:
                final_no_improve += 1
                
            if final_no_improve >= PATIENCE:
                print(f"Early stopping final training at epoch {epoch}.")
                break
                
        # ==========================================================
        # 8. ONE-SHOT FINAL TEST EVALUATION
        # ==========================================================
        print("\n" + "=" * 70)
        print("ONE-SHOT FINAL TEST EVALUATION (ON UNTOUCHED TEST SET)")
        print("=" * 70)
        
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        final_model.load_state_dict(checkpoint["model_state_dict"])
        
        t_loss, t_acc, t_prec, t_rec, t_f1, t_macro_f1, t_targets, t_preds = evaluate(
            final_model, final_test_loader, criterion
        )
        
        print(f"Best Validation Macro-F1 : {best_final_val_f1:.4f} (Epoch {best_final_epoch})")
        print(f"Test Accuracy            : {t_acc:.4f}")
        print(f"Test Weighted F1         : {t_f1:.4f}")
        print(f"Test Macro-F1            : {t_macro_f1:.4f}\n")
        
        print("Classification Report:")
        print(classification_report(t_targets, t_preds, labels=classes, zero_division=0))
        
        cm = confusion_matrix(t_targets, t_preds, labels=classes)
        print("Confusion Matrix:")
        print(cm)
        
        mlflow.log_metrics({
            "best_val_macro_f1": best_final_val_f1,
            "test_accuracy": t_acc,
            "test_weighted_f1": t_f1,
            "test_macro_f1": t_macro_f1,
        })
        
        np.save(os.path.join(MODEL_DIR, "confusion_matrix_optuna_e3b.npy"), cm)
        mlflow.log_artifact(BEST_MODEL_PATH)
        mlflow.log_artifact(os.path.join(MODEL_DIR, "confusion_matrix_optuna_e3b.npy"))
        
    print("\n" + "=" * 70)
    print("ALL DONE - MODEL SAVED TO:")
    print(BEST_MODEL_PATH)
    print("=" * 70)

if __name__ == "__main__":
    main()
