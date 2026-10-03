import os
import random
import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import optuna

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

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
# 1. PATHS & CONFIG
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data processing", "data", "processed"))

TRAIN_PATH = os.path.join(DATA_DIR, "train_final.npz")
VAL_PATH = os.path.join(DATA_DIR, "val_final.npz")
TEST_PATH = os.path.join(DATA_DIR, "test_final.npz")

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_optuna_best.pt")

MLFLOW_DB = os.path.abspath(os.path.join(BASE_DIR, "mlflow.db"))
MLFLOW_URI = "sqlite:///" + MLFLOW_DB.replace("\\", "/")

N_TRIALS = 15
SEARCH_EPOCHS = 15
FINAL_EPOCHS = 50
BATCH_SIZE = 128
PATIENCE = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("OCP FAULT DETECTION - 1D CNN OPTUNA + MLFLOW")
print("=" * 70)
print(f"Device: {DEVICE}\n")

# ==========================================================
# 2. LOAD DATA
# ==========================================================
print("Loading data...")
train = np.load(TRAIN_PATH)
val = np.load(VAL_PATH)
test = np.load(TEST_PATH)

X_train = train["X"].astype(np.float32)
y_train = train["y"].astype(np.int64)
X_val = val["X"].astype(np.float32)
y_val = val["y"].astype(np.int64)
X_test = test["X"].astype(np.float32)
y_test = test["y"].astype(np.int64)

timesteps = X_train.shape[1]
n_features = X_train.shape[2]
classes = np.unique(y_train)
num_classes = len(classes)

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================================
# 3. CLASS WEIGHTS
# ==========================================================
beta = 0.999
class_counts = np.bincount(y_train, minlength=len(classes)).astype(float)
class_counts = np.where(class_counts == 0, 1e-6, class_counts)
effective_num = 1.0 - np.power(beta, class_counts)
weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
weights = weights / np.sum(weights) * len(classes)
weights = np.where(class_counts <= 1e-6, 0.0, weights)
class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

# ==========================================================
# 4. MODEL DEFINITION (1D CNN)
# ==========================================================
class CNN1DClassifier(nn.Module):
    def __init__(self, input_size, num_classes, conv1_out, conv2_out, kernel_size, dropout):
        super().__init__()
        # Input shape: (batch, timesteps, features) -> permute in forward to (batch, features, timesteps)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=conv1_out, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2) # 20 -> 10
        
        self.conv2 = nn.Conv1d(in_channels=conv1_out, out_channels=conv2_out, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2) # 10 -> 5
        
        self.flatten = nn.Flatten()
        # After two MaxPool1d(2), the sequence length is reduced from 20 to 5
        self.fc1 = nn.Linear(conv2_out * 5, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x comes in as (batch, 20, 11). We need it as (batch, 11, 20) for Conv1d
        x = x.permute(0, 2, 1) 
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

# ==========================================================
# 5. TRAINING & EVALUATION LOGIC
# ==========================================================
def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_predictions, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
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
    _, _, macro_f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average="macro", zero_division=0)
    return avg_loss, accuracy, macro_f1, all_targets, all_predictions

def train_and_evaluate(model, params, epochs, trial=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_f1 = -np.inf
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_loss, val_acc, val_macro_f1, _, _ = evaluate_model(model, val_loader, criterion)
        scheduler.step(val_macro_f1)

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if trial is not None:
            trial.report(val_macro_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if epochs_no_improve >= PATIENCE:
            break

    return best_val_f1, best_state_dict

# ==========================================================
# 6. OPTUNA OBJECTIVE
# ==========================================================
def objective(trial):
    with mlflow.start_run(nested=True):
        params = {
            "conv1_out": trial.suggest_categorical("conv1_out", [32, 64, 128]),
            "conv2_out": trial.suggest_categorical("conv2_out", [64, 128, 256]),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        }
        mlflow.log_params(params)
        
        model = CNN1DClassifier(
            input_size=n_features,
            num_classes=num_classes,
            conv1_out=params["conv1_out"],
            conv2_out=params["conv2_out"],
            kernel_size=params["kernel_size"],
            dropout=params["dropout"]
        ).to(DEVICE)
        
        best_val_f1, _ = train_and_evaluate(model, params, SEARCH_EPOCHS, trial)
        mlflow.log_metric("best_val_macro_f1", best_val_f1)
    return best_val_f1

# ==========================================================
# 7. MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("PowerGrid_Optuna_HPT_CNN")

    with mlflow.start_run(run_name="Optuna_Study_CNN_Parent"):
        print("Starting Optuna Hyperparameter Search for 1D CNN...")
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=N_TRIALS)

        print("\n" + "=" * 70)
        print("OPTUNA SEARCH COMPLETE")
        print("=" * 70)
        print(f"Best Trial Value (Macro-F1): {study.best_trial.value:.4f}")
        print("Best Params:")
        best_params = study.best_trial.params
        for k, v in best_params.items():
            print(f"  {k}: {v}")

    with mlflow.start_run(run_name="CNN_Final_Best_Model"):
        mlflow.log_params({
            "model": "1D_CNN",
            "input_features": n_features,
            "timesteps": timesteps,
            "num_classes": num_classes,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "class_weighting": "effective_samples_beta_0.999",
            "seed": SEED,
            **best_params
        })

        print("\n" + "=" * 70)
        print("TRAINING FINAL CNN MODEL WITH BEST PARAMETERS")
        print("=" * 70)

        final_model = CNN1DClassifier(
            input_size=n_features,
            num_classes=num_classes,
            conv1_out=best_params["conv1_out"],
            conv2_out=best_params["conv2_out"],
            kernel_size=best_params["kernel_size"],
            dropout=best_params["dropout"]
        ).to(DEVICE)

        final_val_f1, final_state_dict = train_and_evaluate(final_model, best_params, FINAL_EPOCHS)
        print(f"Final Model Validation Macro-F1: {final_val_f1:.4f}")

        torch.save({
            "model_state_dict": final_state_dict,
            "params": best_params,
            "val_macro_f1": final_val_f1,
        }, MODEL_PATH)
        print(f"PyTorch model saved to: {MODEL_PATH}")

        final_model.load_state_dict(final_state_dict)
        
        print("\n" + "=" * 70)
        print("TEST EVALUATION")
        print("=" * 70)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        test_loss, test_acc, test_macro_f1, test_targets, test_preds = evaluate_model(final_model, test_loader, criterion)
        
        print(f"Test Loss      : {test_loss:.4f}")
        print(f"Test Accuracy  : {test_acc:.4f}")
        print(f"Test Macro-F1  : {test_macro_f1:.4f}\n")

        print("CLASSIFICATION REPORT")
        print(classification_report(test_targets, test_preds, labels=classes, zero_division=0))

        cm = confusion_matrix(test_targets, test_preds, labels=classes)
        print("CONFUSION MATRIX")
        print(cm)

        mlflow.log_metrics({
            "final_val_macro_f1": final_val_f1,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_macro_f1": test_macro_f1
        })

        np.save(os.path.join(MODEL_DIR, "cnn_confusion_matrix.npy"), cm)
        mlflow.log_artifact(MODEL_PATH, artifact_path="saved_model")
        
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)