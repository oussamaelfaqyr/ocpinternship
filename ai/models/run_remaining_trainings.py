import subprocess
import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

scripts = [
    ("GRU E4-B (Attentive GRU + Focal Loss)", "train_e4b.py"),
    ("LSTM Champion (6-Class)", "train_lstm.py"),
    ("1D CNN Champion (6-Class)", "train_cnn.py"),
    ("GRU E5-A (Current Dynamics)", "train_e5a.py"),
]

print("=" * 70)
print("STARTING SEQUENTIAL TRAINING OF REMAINING MODELS (6 CLASSES)")
print("=" * 70)

for name, script in scripts:
    script_path = os.path.join(BASE_DIR, script)
    print(f"\n>>> [STARTING] {name} ({script}) at {time.strftime('%H:%M:%S')}")
    t0 = time.time()
    res = subprocess.run([sys.executable, script_path])
    elapsed = time.time() - t0
    if res.returncode == 0:
        print(f">>> [SUCCESS] {name} finished in {elapsed/60:.2f} minutes.")
    else:
        print(f">>> [FAILED] {name} exited with code {res.returncode}.")

print("\n" + "=" * 70)
print("ALL REMAINING TRAININGS COMPLETED")
print("=" * 70)
