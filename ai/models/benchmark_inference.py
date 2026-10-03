"""
======================================================================
OCP FAULT DETECTION - INFERENCE BENCHMARK
======================================================================

Compare:
    - LSTM Optuna best
    - GRU Optuna best
    - CNN Optuna best

Measures:
    - Number of parameters
    - Mean latency
    - Median latency
    - P95 latency
    - Min latency
    - Max latency
    - Throughput

Hardware:
    CPU

Input:
    Sequence length = 20
    Number of features = 22
    Number of classes = 8
======================================================================
"""

from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "saved_models"

LSTM_PATH = MODEL_DIR / "lstm_optuna_best.pt"
GRU_PATH = MODEL_DIR / "gru_optuna_best.pt"
CNN_PATH = MODEL_DIR / "cnn_optuna_best.pt"

DEVICE = torch.device("cpu")

# IMPORTANT:
# The checkpoints show input_size = 22.
SEQ_LEN = 20
INPUT_SIZE = 22
NUM_CLASSES = 8

# Benchmark configuration
N_SAMPLES = 1000
WARMUP_RUNS = 100
LATENCY_RUNS = 1000

# Batch size used for throughput
BATCH_SIZE = 1000


# ======================================================================
# LSTM MODEL
# ======================================================================

class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        # Last timestep
        last_output = output[:, -1, :]

        return self.fc(last_output)


# ======================================================================
# GRU MODEL
# ======================================================================

class GRUClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        output, _ = self.gru(x)

        # Last timestep
        last_output = output[:, -1, :]

        return self.fc(last_output)


# ======================================================================
# CNN MODEL
# ======================================================================

class CNNClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        conv1_out,
        conv2_out,
        kernel_size,
        num_classes,
        dropout
    ):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=conv1_out,
            kernel_size=kernel_size
        )

        self.conv2 = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        # --------------------------------------------------------------
        # The checkpoint has:
        #
        # fc1.weight = (64, 320)
        #
        # Therefore fc1 receives 320 values:
        #
        # 64 channels × 5 temporal positions = 320
        #
        # Adaptive pooling guarantees exactly 5 positions.
        # --------------------------------------------------------------

        self.adaptive_pool = nn.AdaptiveAvgPool1d(5)

        self.fc1 = nn.Linear(
            conv2_out * 5,
            64
        )

        self.fc2 = nn.Linear(
            64,
            num_classes
        )

    def forward(self, x):

        # Input:
        # (batch, sequence, features)
        #
        # Conv1d expects:
        # (batch, channels, sequence)

        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.adaptive_pool(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.dropout(
            self.fc1(x)
        )

        x = self.relu(x)

        x = self.fc2(x)

        return x


# ======================================================================
# LOAD LSTM
# ======================================================================

def load_lstm():

    print("\nLoading LSTM checkpoint...")

    checkpoint = torch.load(
        LSTM_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    params = checkpoint["params"]

    print("LSTM parameters:")
    print(f"  hidden_size : {params['hidden_size']}")
    print(f"  num_layers  : {params['num_layers']}")
    print(f"  dropout     : {params['dropout']:.6f}")
    print(f"  val macro F1: {checkpoint['val_macro_f1']:.6f}")

    model = LSTMClassifier(
        input_size=INPUT_SIZE,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        num_classes=NUM_CLASSES,
        dropout=params["dropout"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


# ======================================================================
# LOAD GRU
# ======================================================================

def load_gru():

    print("\nLoading GRU checkpoint...")

    checkpoint = torch.load(
        GRU_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    print("GRU parameters:")
    print(f"  input_size  : {checkpoint['input_size']}")
    print(f"  hidden_size : {checkpoint['hidden_size']}")
    print(f"  num_layers  : {checkpoint['num_layers']}")
    print(f"  dropout     : {checkpoint['dropout']:.6f}")
    print(f"  val macro F1: {checkpoint['val_macro_f1']:.6f}")

    model = GRUClassifier(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint["dropout"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


# ======================================================================
# LOAD CNN
# ======================================================================

def load_cnn():

    print("\nLoading CNN checkpoint...")

    checkpoint = torch.load(
        CNN_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    params = checkpoint["params"]

    print("CNN parameters:")
    print(f"  conv1_out   : {params['conv1_out']}")
    print(f"  conv2_out   : {params['conv2_out']}")
    print(f"  kernel_size : {params['kernel_size']}")
    print(f"  dropout     : {params['dropout']:.6f}")
    print(f"  val macro F1: {checkpoint['val_macro_f1']:.6f}")

    model = CNNClassifier(
        input_size=INPUT_SIZE,
        conv1_out=params["conv1_out"],
        conv2_out=params["conv2_out"],
        kernel_size=params["kernel_size"],
        num_classes=NUM_CLASSES,
        dropout=params["dropout"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


# ======================================================================
# COUNT PARAMETERS
# ======================================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ======================================================================
# WARM-UP
# ======================================================================

@torch.inference_mode()
def warmup(model, x):

    for _ in range(WARMUP_RUNS):

        _ = model(x)


# ======================================================================
# TEST FORWARD PASS
# ======================================================================

@torch.inference_mode()
def test_forward(model, x):

    output = model(x)

    return output


# ======================================================================
# BATCH THROUGHPUT
# ======================================================================

@torch.inference_mode()
def benchmark_throughput(model, x):

    warmup(
        model,
        x
    )

    start = time.perf_counter()

    _ = model(x)

    end = time.perf_counter()

    total_time = end - start

    throughput = (
        x.shape[0] / total_time
    )

    latency_per_sample = (
        total_time / x.shape[0]
    ) * 1000.0

    return (
        total_time,
        throughput,
        latency_per_sample
    )


# ======================================================================
# SINGLE-SAMPLE LATENCY
# ======================================================================

@torch.inference_mode()
def benchmark_latency(model, x):

    warmup(
        model,
        x
    )

    latencies = []

    for _ in range(LATENCY_RUNS):

        start = time.perf_counter()

        _ = model(x)

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000.0

        latencies.append(
            latency_ms
        )

    latencies = np.asarray(
        latencies
    )

    return {
        "mean": float(np.mean(latencies)),
        "median": float(np.median(latencies)),
        "p95": float(np.percentile(latencies, 95)),
        "min": float(np.min(latencies)),
        "max": float(np.max(latencies))
    }


# ======================================================================
# BENCHMARK ONE MODEL
# ======================================================================

def benchmark_model(
    name,
    model,
    x_batch,
    x_single
):

    print("\n")
    print("=" * 70)
    print(f"BENCHMARKING {name}")
    print("=" * 70)

    # --------------------------------------------------------------
    # Parameters
    # --------------------------------------------------------------

    parameters = count_parameters(
        model
    )

    print(
        f"\nTrainable parameters: "
        f"{parameters:,}"
    )

    # --------------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------------

    print(
        "\nTesting forward pass..."
    )

    output = test_forward(
        model,
        x_single
    )

    print(
        f"Input shape : "
        f"{tuple(x_single.shape)}"
    )

    print(
        f"Output shape: "
        f"{tuple(output.shape)}"
    )

    # --------------------------------------------------------------
    # Throughput
    # --------------------------------------------------------------

    print(
        "\nRunning throughput benchmark..."
    )

    (
        total_time,
        throughput,
        batch_latency
    ) = benchmark_throughput(
        model,
        x_batch
    )

    # --------------------------------------------------------------
    # Single sample latency
    # --------------------------------------------------------------

    print(
        "Running single-sample latency benchmark..."
    )

    latency = benchmark_latency(
        model,
        x_single
    )

    # --------------------------------------------------------------
    # Results
    # --------------------------------------------------------------

    result = {

        "parameters": parameters,

        "total_time": total_time,

        "throughput": throughput,

        "batch_latency": batch_latency,

        "mean_latency": latency["mean"],

        "median_latency": latency["median"],

        "p95_latency": latency["p95"],

        "min_latency": latency["min"],

        "max_latency": latency["max"]
    }

    print("\nResults:")
    print("-" * 50)

    print(
        f"Total batch time : "
        f"{total_time:.6f} s"
    )

    print(
        f"Throughput       : "
        f"{throughput:.2f} samples/s"
    )

    print(
        f"Batch latency    : "
        f"{batch_latency:.4f} ms/sample"
    )

    print(
        f"Mean latency     : "
        f"{latency['mean']:.4f} ms"
    )

    print(
        f"Median latency   : "
        f"{latency['median']:.4f} ms"
    )

    print(
        f"P95 latency      : "
        f"{latency['p95']:.4f} ms"
    )

    print(
        f"Min latency      : "
        f"{latency['min']:.4f} ms"
    )

    print(
        f"Max latency      : "
        f"{latency['max']:.4f} ms"
    )

    return result


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print(
        "OCP FAULT DETECTION - INFERENCE BENCHMARK"
    )
    print("=" * 70)

    print(
        f"\nDevice       : {DEVICE}"
    )

    print(
        f"Sequence len : {SEQ_LEN}"
    )

    print(
        f"Features     : {INPUT_SIZE}"
    )

    print(
        f"Classes      : {NUM_CLASSES}"
    )

    print(
        f"Samples      : {N_SAMPLES}"
    )

    print(
        f"Warm-up      : {WARMUP_RUNS}"
    )

    print(
        f"Latency runs : {LATENCY_RUNS}"
    )

    # ==================================================================
    # CREATE INPUT
    # ==================================================================

    print(
        "\nCreating benchmark input..."
    )

    torch.manual_seed(42)

    x_batch = torch.randn(
        BATCH_SIZE,
        SEQ_LEN,
        INPUT_SIZE,
        dtype=torch.float32,
        device=DEVICE
    )

    x_single = x_batch[0:1]

    print(
        f"Batch input: "
        f"{tuple(x_batch.shape)}"
    )

    print(
        f"Single input: "
        f"{tuple(x_single.shape)}"
    )

    # ==================================================================
    # LOAD MODELS
    # ==================================================================

    models = {}

    try:

        models["LSTM"] = load_lstm()

    except Exception as e:

        print(
            "\nERROR loading LSTM:"
        )

        print(
            repr(e)
        )

    try:

        models["GRU"] = load_gru()

    except Exception as e:

        print(
            "\nERROR loading GRU:"
        )

        print(
            repr(e)
        )

    try:

        models["CNN"] = load_cnn()

    except Exception as e:

        print(
            "\nERROR loading CNN:"
        )

        print(
            repr(e)
        )

    # ==================================================================
    # BENCHMARK
    # ==================================================================

    results = {}

    for name, model in models.items():

        try:

            results[name] = benchmark_model(
                name,
                model,
                x_batch,
                x_single
            )

        except Exception as e:

            print(
                f"\nERROR benchmarking {name}:"
            )

            print(
                repr(e)
            )

            results[name] = None

    # ==================================================================
    # FINAL TABLE
    # ==================================================================

    print("\n\n")

    print("=" * 105)
    print("FINAL COMPARISON")
    print("=" * 105)

    print(
        f"{'Model':<10}"
        f"{'Params':>15}"
        f"{'Mean ms':>15}"
        f"{'Median ms':>15}"
        f"{'P95 ms':>15}"
        f"{'Throughput':>20}"
    )

    print("-" * 105)

    for name in ["LSTM", "GRU", "CNN"]:

        result = results.get(name)

        if result is None:

            print(
                f"{name:<10}"
                f"{'ERROR':>15}"
            )

            continue

        print(
            f"{name:<10}"
            f"{result['parameters']:>15,}"
            f"{result['mean_latency']:>15.4f}"
            f"{result['median_latency']:>15.4f}"
            f"{result['p95_latency']:>15.4f}"
            f"{result['throughput']:>17.2f}/s"
        )

    # ==================================================================
    # FASTEST MODEL
    # ==================================================================

    valid_results = {
        name: result
        for name, result in results.items()
        if result is not None
    }

    if len(valid_results) > 0:

        fastest_latency = min(
            valid_results.items(),
            key=lambda item:
            item[1]["mean_latency"]
        )

        fastest_throughput = max(
            valid_results.items(),
            key=lambda item:
            item[1]["throughput"]
        )

        print("\n")
        print("=" * 70)

        print(
            f"FASTEST BY LATENCY : "
            f"{fastest_latency[0]}"
        )

        print(
            f"Mean latency       : "
            f"{fastest_latency[1]['mean_latency']:.4f} ms"
        )

        print()

        print(
            f"BEST THROUGHPUT    : "
            f"{fastest_throughput[0]}"
        )

        print(
            f"Throughput         : "
            f"{fastest_throughput[1]['throughput']:.2f} samples/s"
        )

    # ==================================================================
    # VALIDATION INFORMATION
    # ==================================================================

    print("\n")
    print("=" * 70)
    print("MODEL VALIDATION INFORMATION")
    print("=" * 70)

    print(
        "\nLSTM validation Macro-F1:"
    )

    lstm_checkpoint = torch.load(
        LSTM_PATH,
        map_location="cpu",
        weights_only=False
    )

    print(
        f"{lstm_checkpoint['val_macro_f1']:.6f}"
    )

    print(
        "\nGRU validation Macro-F1:"
    )

    gru_checkpoint = torch.load(
        GRU_PATH,
        map_location="cpu",
        weights_only=False
    )

    print(
        f"{gru_checkpoint['val_macro_f1']:.6f}"
    )

    print(
        "\nCNN validation Macro-F1:"
    )

    cnn_checkpoint = torch.load(
        CNN_PATH,
        map_location="cpu",
        weights_only=False
    )

    print(
        f"{cnn_checkpoint['val_macro_f1']:.6f}"
    )

    print("\n")
    print("=" * 70)
    print("BENCHMARK COMPLETED")
    print("=" * 70)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    main()