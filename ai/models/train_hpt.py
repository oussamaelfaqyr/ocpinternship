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

# Suppress warnings for cleaner output
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
# 1. PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data processing", "data", "processed"))

TRAIN_PATH = os.path.join(DATA_DIR, "train_final.npz")
VAL_PATH = os.path.join(DATA_DIR, "val_final.npz")
TEST_PATH = os.path.join(DATA_DIR, "test_final.npz")

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "gru_optuna_best.pt")

MLFLOW_DB = os.path.abspath(os.path.join(BASE_DIR, "mlflow.db"))
MLFLOW_URI = "sqlite:///" + MLFLOW_DB.replace("\\", "/")

# ==========================================================
# 2. HYPERPARAMETER SEARCH CONFIG
# ==========================================================

N_TRIALS = 15           # Number of Optuna combinations to try
SEARCH_EPOCHS = 15      # Reduced epochs for fast Optuna search
FINAL_EPOCHS = 50       # Full epochs for the final best model
BATCH_SIZE = 128
PATIENCE = 7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("OCP ELECTRICAL FAULT DETECTION - GRU OPTUNA + MLFLOW")
print("=" * 70)
print(f"Device: {DEVICE}\n")

# ==========================================================
# 3. LOAD DATA
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())

# ==========================================================
# 4. CLASS WEIGHTS
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
# 5. MODEL DEFINITION
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
        output, hidden = self.gru(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        return logits

# ==========================================================
# 6. TRAINING & EVALUATION LOGIC
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
    _, _, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average="macro", zero_division=0
    )
    return avg_loss, accuracy, macro_f1, all_targets, all_predictions


def train_and_evaluate(params, epochs, trial=None):
    model = GRUClassifier(
        input_size=n_features,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        num_classes=num_classes,
        dropout=params["dropout"]
    ).to(DEVICE)

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

        # Optuna Pruning Hook
        if trial is not None:
            trial.report(val_macro_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if epochs_no_improve >= PATIENCE:
            break

    return best_val_f1, best_state_dict

# ==========================================================
# 7. OPTUNA OBJECTIVE
# ==========================================================

def objective(trial):
    # Nested MLflow run for each Optuna Trial
    with mlflow.start_run(nested=True):
        params = {
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        }
        mlflow.log_params(params)
        
        best_val_f1, _ = train_and_evaluate(params, SEARCH_EPOCHS, trial)
        mlflow.log_metric("best_val_macro_f1", best_val_f1)
        
    return best_val_f1

# ==========================================================
# 8. MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("PowerGrid_Optuna_HPT")

    # Parent MLflow Run for the entire Optuna Study
    with mlflow.start_run(run_name="Optuna_Study_Parent"):
        print("Starting Optuna Hyperparameter Search...")
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        
        # Removed the MLflowCallback to prevent the "already active" exception
        study.optimize(objective, n_trials=N_TRIALS)

        print("\n" + "=" * 70)
        print("OPTUNA SEARCH COMPLETE")
        print("=" * 70)
        print(f"Best Trial Value (Macro-F1): {study.best_trial.value:.4f}")
        print("Best Params:")
        best_params = study.best_trial.params
        for k, v in best_params.items():
            print(f"  {k}: {v}")

    # Start a completely separate top-level run for the final best model
    with mlflow.start_run(run_name="GRU_Final_Best_Model"):
        mlflow.log_params({
            "model": "GRU",
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
        print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
        print("=" * 70)

        final_val_f1, final_state_dict = train_and_evaluate(best_params, FINAL_EPOCHS)
        print(f"Final Model Validation Macro-F1: {final_val_f1:.4f}")

        # Save the final best model locally
        torch.save({
            "model_state_dict": final_state_dict,
            "input_size": n_features,
            "hidden_size": best_params["hidden_size"],
            "num_layers": best_params["num_layers"],
            "num_classes": num_classes,
            "dropout": best_params["dropout"],
            "classes": classes.tolist(),
            "val_macro_f1": final_val_f1,
        }, MODEL_PATH)
        print(f"PyTorch model saved to: {MODEL_PATH}")

        # Load best weights for test evaluation
        model = GRUClassifier(
            input_size=n_features,
            hidden_size=best_params["hidden_size"],
            num_layers=best_params["num_layers"],
            num_classes=num_classes,
            dropout=best_params["dropout"]
        ).to(DEVICE)
        model.load_state_dict(final_state_dict)
        
        # Evaluate on Test Set
        print("\n" + "=" * 70)
        print("TEST EVALUATION")
        print("=" * 70)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        test_loss, test_acc, test_macro_f1, test_targets, test_preds = evaluate_model(model, test_loader, criterion)
        
        print(f"Test Loss      : {test_loss:.4f}")
        print(f"Test Accuracy  : {test_acc:.4f}")
        print(f"Test Macro-F1  : {test_macro_f1:.4f}\n")

        print("CLASSIFICATION REPORT")
        print(classification_report(test_targets, test_preds, labels=classes, zero_division=0))

        cm = confusion_matrix(test_targets, test_preds, labels=classes)
        print("CONFUSION MATRIX")
        print(cm)

        # Log final metrics and artifacts to MLflow
        mlflow.log_metrics({
            "final_val_macro_f1": final_val_f1,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_macro_f1": test_macro_f1
        })

        class_mapping = {int(c): int(c) for c in classes}
        joblib.dump(class_mapping, os.path.join(MODEL_DIR, "class_mapping.pkl"))
        np.save(os.path.join(MODEL_DIR, "confusion_matrix.npy"), cm)

        mlflow.log_artifact(MODEL_PATH, artifact_path="saved_model")
        mlflow.log_artifact(os.path.join(MODEL_DIR, "confusion_matrix.npy"), artifact_path="evaluation")
        
        try:
            mlflow.pytorch.log_model(model, artifact_path="pytorch_model")
        except Exception as e:
            print(f"Could not log PyTorch model to MLflow: {e}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"Run 'mlflow ui --backend-store-uri {MLFLOW_URI}' to view results.")
    print("=" * 70)