from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ROOT_DATASET_DIR = BASE_DIR.parent.parent / "dataset"
DATASET_DIR = ROOT_DATASET_DIR if (ROOT_DATASET_DIR / "train.csv").exists() else (BASE_DIR.parent / "dataset")
OUTPUT_DIR = BASE_DIR / "data" / "processed"

TRAIN_RAW = DATASET_DIR / "train.csv"

WINDOW_SIZE = 20
STRIDE = 10

VAL_FRACTION = 0.50

EXCLUDED_EVAL_LABELS = [
    "breaker_trip",
    "over_frequency",
    "under_frequency",
]

# ADDED "PF" back into numeric features, since we fixed the simulator 
# to generate dynamic, physics-based Power Factors.
NUMERIC_FEATURES = [
    "V_hv_kV",
    "V_lv_kV",
    "I_hv_A",
    "P_MW",
    "Q_Mvar",
    "S_MVA",
    "PF",               # <-- ADDED BACK
    "loading_pct",
    "freq_Hz",
    "delta_freq_Hz",
    "delta_V_hv_kV",
    "V_HV_pu",          # <-- E1 Feature
    "V_LV_pu",          # <-- E1 Feature
    "V_ratio",          # <-- E1 Feature
    "delta_V_pu",       # <-- E1 Feature
    "P_ratio",          # <-- E2 Feature
    "P_delta_ratio",    # <-- E2 Feature
    "I_ratio",          # <-- E2 Feature
    "S_ratio",          # <-- E2 Feature
    "delta_I_hv",       # <-- E2 Feature
    "abs_delta_I_hv",   # <-- E2 Feature
    "delta_P_MW",       # <-- E2 Feature
    "delta_S_MVA",      # <-- E2 Feature
    "I2t_pu",           # <-- E2 Feature
]

# Removed "PF" from drop columns
DROP_COLUMNS = [
    "relay_status",
    "breaker_status",
    "lifecycle_stage",
    "fault_location",
    "equip_status",
    "offline_subs_fraction",
]

REMAP_LABELS = {}

PF_BOUNDS = (-1.0, 1.0)
FREQ_BOUNDS = (45.0, 55.0)

S_CONSISTENCY_TOLERANCE = 0.15   # Slightly increased tolerance for dynamic float math
PF_CONSISTENCY_TOLERANCE = 0.15  # Slightly increased tolerance for dynamic float math


# ============================================================
# UTILITIES
# ============================================================

def ensure_columns(df, required, name):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"\n{name}: missing required columns:\n"
            f"{missing}\n\nAvailable columns:\n{list(df.columns)}"
        )

def print_labels(df, name):
    print(f"\n[{name}] label distribution:")
    print(df["label"].value_counts().sort_index().to_string())


# ============================================================
# EPISODES
# ============================================================

def assign_episode_ids(df):
    if "episode_id" in df.columns:
        return df.copy()
    df = df.sort_values(["substation", "time_s"]).reset_index(drop=True).copy()
    dt = df.groupby("substation")["time_s"].diff()
    expected_dt = df.groupby("substation")["time_s"].diff().dropna().median()
    if pd.isna(expected_dt):
        expected_dt = 1.0
    previous_label = df.groupby("substation")["label"].shift()
    new_episode = dt.isna() | (dt != expected_dt) | (df["label"] != previous_label)
    df["episode_id"] = new_episode.astype(np.int64).cumsum()
    return df

def episode_level_split(df, train_frac=0.70, val_frac=0.15, seed=42):
    print("\n[EPISODE SPLIT]")
    df = assign_episode_ids(df)
    rng = np.random.RandomState(seed)
    
    # Determine primary episode label for stratification
    if "event_label" in df.columns:
        episodes = df.groupby("episode_id")["event_label"].first().reset_index()
        episodes.rename(columns={"event_label": "split_label"}, inplace=True)
    else:
        def find_label(s):
            nn = s[s != "normal"]
            return nn.iloc[0] if len(nn) > 0 else "normal"
        episodes = df.groupby("episode_id")["label"].agg(find_label).reset_index()
        episodes.rename(columns={"label": "split_label"}, inplace=True)

    train_ids, val_ids, test_ids = [], [], []

    for label, group in episodes.groupby("split_label"):
        ids = group["episode_id"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))

        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))
        elif n == 2:
            n_train = 1
            n_val = 0
            # 1 goes to test
        else:
            n_train = 1
            n_val = 0

        train_class_ids = ids[:n_train]
        val_class_ids = ids[n_train : n_train + n_val]
        test_class_ids = ids[n_train + n_val :]

        train_ids.extend(train_class_ids)
        val_ids.extend(val_class_ids)
        test_ids.extend(test_class_ids)

        print(
            f"  {label:25s}: {n:3d} episodes -> "
            f"{len(train_class_ids):2d} train / "
            f"{len(val_class_ids):2d} val / "
            f"{len(test_class_ids):2d} test"
        )

    train_ids, val_ids, test_ids = set(train_ids), set(val_ids), set(test_ids)
    if train_ids & val_ids: raise RuntimeError("Episode leakage between train and validation.")
    if train_ids & test_ids: raise RuntimeError("Episode leakage between train and test.")
    if val_ids & test_ids: raise RuntimeError("Episode leakage between validation and test.")

    train_split = df[df["episode_id"].isin(train_ids)].copy()
    val_split = df[df["episode_id"].isin(val_ids)].copy()
    test_split = df[df["episode_id"].isin(test_ids)].copy()

    print(f"\n  Train episodes: {len(train_ids)}")
    print(f"  Val episodes  : {len(val_ids)}")
    print(f"  Test episodes : {len(test_ids)}")
    print(f"\n  Train rows: {len(train_split)}")
    print(f"  Val rows  : {len(val_split)}")
    print(f"  Test rows : {len(test_split)}")

    return train_split, val_split, test_split


# ============================================================
# CLEANING
# ============================================================

def clean_dataframe(df, name):
    print(f"\n[{name}] Cleaning")
    print("  raw shape:", df.shape)
    before = len(df)
    df = df.drop_duplicates()
    print("  exact duplicates removed:", before - len(df))

    pf_invalid = ~df["PF"].between(*PF_BOUNDS) & df["PF"].notna()
    loading_invalid = df["loading_pct"] < 0
    voltage_invalid = df["V_hv_kV"] < 0
    current_invalid = df["I_hv_A"] < 0
    frequency_invalid = ~df["freq_Hz"].between(*FREQ_BOUNDS) & df["freq_Hz"].notna()

    print("  PF outside [-1,1]:", int(pf_invalid.sum()))
    print("  loading_pct < 0:", int(loading_invalid.sum()))
    print("  V_hv_kV < 0:", int(voltage_invalid.sum()))
    print("  I_hv_A < 0:", int(current_invalid.sum()))
    print("  frequency outside bounds:", int(frequency_invalid.sum()))

    # Ensure episode_id exists for safe groupby without leakage
    if "episode_id" not in df.columns:
        df = assign_episode_ids(df)
        
    work = df.sort_values(["episode_id", "substation", "time_s"]).reset_index(drop=True).copy()
    
    missing_before = int(work[NUMERIC_FEATURES].isna().sum().sum())
    
    # Linear interpolation inside each (episode, substation) series
    work[NUMERIC_FEATURES] = work.groupby(["episode_id", "substation"])[NUMERIC_FEATURES].transform(
        lambda g: g.interpolate(method="linear", limit_area="inside")
    )
    
    for column in NUMERIC_FEATURES:
        work[column] = work.groupby(["episode_id", "substation"])[column].ffill().bfill()

    missing_after = int(work[NUMERIC_FEATURES].isna().sum().sum())
    print(f"  numeric NaNs: {missing_before} -> {missing_after}")

    return work


# ============================================================
# VALIDATION
# ============================================================

def validate_dataframe(df, name):
    print(f"\n[{name}] Validation")
    s_calculated = np.sqrt(df["P_MW"] ** 2 + df["Q_Mvar"] ** 2)
    s_mismatch = (df["S_MVA"] - s_calculated).abs() > S_CONSISTENCY_TOLERANCE
    print("  S != sqrt(P^2+Q^2):", int(s_mismatch.sum()))

    s_safe = df["S_MVA"].replace(0, np.nan)
    pf_calculated = (df["P_MW"] / s_safe).clip(-1, 1)
    pf_mismatch = ((df["PF"] - pf_calculated).abs() > PF_CONSISTENCY_TOLERANCE).fillna(False)
    print("  PF != P/S:", int(pf_mismatch.sum()), f"({pf_mismatch.mean() * 100:.2f}%)")

    if "breaker_status" in df.columns and "event_label" in df.columns:
        # Breaker is still OPEN after fault -> feeder physically isolated -> keep fault label
        open_mask = (df["label"] == "normal") & (df["breaker_status"] == "OPEN")
        df.loc[open_mask, "label"] = df.loc[open_mask, "event_label"]
        print(f"  Relabeled {open_mask.sum()} 'normal+OPEN' rows with their event_label")
    df["is_transition_row"] = np.int8(0)

    before = len(df)
    df = df.dropna(subset=NUMERIC_FEATURES).copy()
    print("  rows removed because of NaNs:", before - len(df))
    print_labels(df, name)
    return df


# ============================================================
# FEATURE SELECTION
# ============================================================

def select_features(df, name):
    print(f"\n[{name}] Feature selection")
    columns_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    print("  dropping:", columns_to_drop)
    df = df.drop(columns=columns_to_drop)
    keep_columns = ["episode_id", "time_s", "substation"] + NUMERIC_FEATURES + ["is_transition_row", "label"]
    ensure_columns(df, keep_columns, name)
    return df[keep_columns].copy()


# ============================================================
# FREQUENCY CORRECTION
# ============================================================

def fix_frequency_range(df):
    """
    PASSTHROUGH: Frequency rescaling conditionally on target labels caused severe
    data leakage and produced unseen values at inference.
    """
    return df


# ============================================================
# DELTA FEATURES
# ============================================================

def add_delta_features(df):
    df = df.sort_values(["episode_id", "substation", "time_s"]).copy()
    group = df.groupby(["episode_id", "substation"])
    df["delta_freq_Hz"] = group["freq_Hz"].diff().fillna(0.0)
    df["delta_V_hv_kV"] = group["V_hv_kV"].diff().fillna(0.0)
    return df


def add_voltage_features(df):
    df = df.copy()
    eps = 1e-6
    df["V_HV_pu"] = df["V_hv_kV"] / 60.0
    df["V_LV_pu"] = df["V_lv_kV"] / 5.5
    df["V_ratio"] = df["V_LV_pu"] / (df["V_HV_pu"] + eps)
    df["delta_V_pu"] = df["V_LV_pu"] - df["V_HV_pu"]
    return df


def compute_train_baselines(train_df):
    """Compute normal baseline power and current strictly on TRAIN normal rows."""
    normal_train = train_df[train_df["label"] == "normal"]
    baseline_P = normal_train.groupby("substation")["P_MW"].median().to_dict()
    baseline_I = normal_train.groupby("substation")["I_hv_A"].median().to_dict()
    baseline_S = normal_train.groupby("substation")["S_MVA"].median().to_dict()
    return {
        "P": baseline_P,
        "I": baseline_I,
        "S": baseline_S
    }


def apply_power_and_dynamic_features(df, baselines):
    """Apply substation-relative ratios and dynamic features (E2)."""
    df = df.sort_values(["episode_id", "substation", "time_s"]).copy()
    eps = 1e-4

    p_base = df["substation"].map(baselines["P"])
    i_base = df["substation"].map(baselines["I"])
    s_base = df["substation"].map(baselines["S"])

    # Relative ratios
    df["P_ratio"] = df["P_MW"] / (p_base + eps)
    df["P_delta_ratio"] = (df["P_MW"] - p_base) / (p_base + eps)
    df["I_ratio"] = df["I_hv_A"] / (i_base + eps)
    df["S_ratio"] = df["S_MVA"] / (s_base + eps)
    df["I2t_pu"] = (df["I_hv_A"] / (i_base + eps)) ** 2

    # Dynamic variations strictly within the same episode and substation
    group = df.groupby(["episode_id", "substation"])
    df["delta_I_hv"] = group["I_hv_A"].diff().fillna(0.0)
    df["abs_delta_I_hv"] = df["delta_I_hv"].abs()
    df["delta_P_MW"] = group["P_MW"].diff().fillna(0.0)
    df["delta_S_MVA"] = group["S_MVA"].diff().fillna(0.0)

    diff_cols = ["delta_I_hv", "abs_delta_I_hv", "delta_P_MW", "delta_S_MVA"]
    df[diff_cols] = df[diff_cols].fillna(0.0)

    return df


# ============================================================
# WINDOWING
# ============================================================

def label_window(labels):
    fault_labels = [label for label in labels if label != "normal"]
    if fault_labels:
        return fault_labels[-1]
    return "normal"

def make_windows(df, name):
    print(f"\n[{name}] Windowing")

    # Sort strictly by episode, substation, and time_s
    df = df.sort_values(["episode_id", "substation", "time_s"]).reset_index(drop=True)

    all_X, all_y, all_t, all_substations, all_transition, all_episodes = [], [], [], [], [], []

    # Group by BOTH episode_id and substation to ensure each window is a continuous temporal sequence
    grouped = df.groupby(["episode_id", "substation"], sort=False)

    for (ep_id, substation), group in grouped:
        group = group.sort_values("time_s").reset_index(drop=True)
        if len(group) < WINDOW_SIZE:
            continue

        for start in range(0, len(group) - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE
            window = group.iloc[start:end]

            X = window[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
            y = label_window(window["label"].to_numpy())

            all_X.append(X)
            all_y.append(y)
            all_t.append(window["time_s"].iloc[0])
            all_substations.append(substation)
            all_transition.append(float(window["is_transition_row"].mean()))
            all_episodes.append(ep_id)

    if not all_X:
        raise ValueError(
            f"{name}: no windows generated. "
            f"Each episode/substation series must contain at least {WINDOW_SIZE} timesteps."
        )

    X = np.stack(all_X)
    y = np.asarray(all_y)
    t_start = np.asarray(all_t)
    substations = np.asarray(all_substations)
    transition_fraction = np.asarray(all_transition, dtype=np.float32)
    episodes_arr = np.asarray(all_episodes)

    print("  X:", X.shape)
    values, counts = np.unique(y, return_counts=True)
    print("  labels:", dict(zip(values, counts)))

    return {
        "X": X,
        "y": y,
        "t_start": t_start,
        "substation": substations,
        "transition_frac": transition_fraction,
        "episode_id": episodes_arr,
    }


# ============================================================
# DROP UNSEEN EVALUATION CLASSES
# ============================================================

def remove_unseen_windows(windows, train_labels, name):
    unseen = sorted(set(windows["y"].tolist()) - set(train_labels))
    if not unseen:
        return windows

    print(f"\n[{name}] Removing unseen classes:", unseen)
    mask = ~np.isin(windows["y"], unseen)
    result = {k: v[mask] for k, v in windows.items()}
    if len(result["X"]) == 0:
        raise ValueError(f"{name}: no windows remain after removing unseen classes.")
    return result


# ============================================================
# ENCODING
# ============================================================

def fit_encoding(train_windows):
    print("\n[ENCODING] Fit on train only")
    train_substations = sorted(set(train_windows["substation"].tolist()))
    ohe = OneHotEncoder(
        sparse_output=False,
        categories=[train_substations],
        handle_unknown="ignore",
    )
    ohe.fit(np.asarray(train_substations).reshape(-1, 1))

    labels = sorted(set(train_windows["y"].tolist()))
    if "normal" in labels:
        labels.remove("normal")
        labels = ["normal"] + labels
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    print("  substations:", train_substations)
    print("  labels:", label_to_idx)
    return ohe, label_to_idx

def encode_windows(windows, ohe, label_to_idx, name):
    unseen = sorted(set(windows["y"].tolist()) - set(label_to_idx.keys()))
    if unseen:
        raise ValueError(f"{name} contains labels absent from training: {unseen}")

    X_numeric = windows["X"]
    window_size = X_numeric.shape[1]
    substation_encoded = ohe.transform(windows["substation"].reshape(-1, 1))
    substation_encoded = np.repeat(substation_encoded[:, None, :], window_size, axis=1)
    X = np.concatenate([X_numeric, substation_encoded], axis=2)
    y = np.asarray([label_to_idx[label] for label in windows["y"]], dtype=np.int64)

    return {
        "X": X,
        "y": y,
        "t_start": windows["t_start"],
        "substation": windows["substation"],
        "transition_frac": windows["transition_frac"],
        "episode_id": windows["episode_id"],
    }


# ============================================================
# SCALING
# ============================================================

def fit_scaler(train_encoded):
    print("\n[SCALING] Fit on train only")
    n_numeric = len(NUMERIC_FEATURES)
    numeric = train_encoded["X"][:, :, :n_numeric]
    scaler = StandardScaler()
    scaler.fit(numeric.reshape(-1, n_numeric))
    print("  mean:", np.round(scaler.mean_, 3).tolist())
    print("  scale:", np.round(scaler.scale_, 3).tolist())
    return scaler

def apply_scaler(encoded, scaler):
    n_numeric = len(NUMERIC_FEATURES)
    X = encoded["X"]
    numeric = X[:, :, :n_numeric]
    categorical = X[:, :, n_numeric:]
    n, w, f = numeric.shape
    numeric_scaled = scaler.transform(numeric.reshape(-1, f)).reshape(n, w, f)
    X_scaled = np.concatenate([numeric_scaled, categorical], axis=2)
    result = dict(encoded)
    result["X"] = X_scaled.astype(np.float32)
    return result


# ============================================================
# SAVE
# ============================================================

def save_npz(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "X": data["X"],
        "y": data["y"],
        "t_start": data["t_start"],
        "substation": data["substation"],
        "transition_frac": data["transition_frac"],
    }
    if "episode_id" in data:
        save_dict["episode_id"] = data["episode_id"]
    np.savez_compressed(path, **save_dict)
    print("  saved:", path)


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()
    print("=" * 70)
    print("OCP ELECTRICAL FAULT DETECTION PREPROCESSING")
    print("=" * 70)

    if not TRAIN_RAW.exists():
        raise FileNotFoundError(f"Missing dataset:\n{TRAIN_RAW}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("A. LOAD DATASET")
    print("=" * 70)
    raw = pd.read_csv(TRAIN_RAW)
    required_columns = [
        "time_s", "substation", "V_hv_kV", "V_lv_kV", "I_hv_A", "P_MW",
        "Q_Mvar", "S_MVA", "PF", "loading_pct", "freq_Hz", "label"
    ]
    ensure_columns(raw, required_columns, "train.csv")
    print("SOURCE:", raw.shape)

    # Drop entire frequency episodes before split (85% of their rows are nominal — no learnable signal)
    if "event_label" in raw.columns:
        freq_episode_ids = raw.loc[raw["event_label"].isin(["over_frequency", "under_frequency"]), "episode_id"].unique()
        raw = raw[~raw["episode_id"].isin(freq_episode_ids)].copy()
        print(f"  Dropped {len(freq_episode_ids)} frequency episodes -> {raw['episode_id'].nunique()} episodes remaining")

    print("\n" + "=" * 70)
    print("B. EPISODE SPLIT")
    print("=" * 70)
    train_raw, val_raw, test_raw = episode_level_split(raw, train_frac=0.70, val_frac=0.15, seed=42)

    print("\n" + "=" * 70)
    print("C. REMOVE EXCLUDED CLASSES")
    print("=" * 70)
    train_raw = train_raw[~train_raw["label"].isin(EXCLUDED_EVAL_LABELS)].copy()
    val_raw = val_raw[~val_raw["label"].isin(EXCLUDED_EVAL_LABELS)].copy()
    test_raw = test_raw[~test_raw["label"].isin(EXCLUDED_EVAL_LABELS)].copy()

    print("\n" + "=" * 70)
    print("D. FREQUENCY CORRECTION")
    print("=" * 70)
    train_raw = fix_frequency_range(train_raw)
    val_raw = fix_frequency_range(val_raw)
    test_raw = fix_frequency_range(test_raw)

    print("\n" + "=" * 70)
    print("E. DELTA, VOLTAGE & POWER DYNAMIC FEATURES (E1 & E2)")
    print("=" * 70)
    train_raw = add_delta_features(train_raw)
    val_raw = add_delta_features(val_raw)
    test_raw = add_delta_features(test_raw)

    train_raw = add_voltage_features(train_raw)
    val_raw = add_voltage_features(val_raw)
    test_raw = add_voltage_features(test_raw)

    # E2: Compute substation normal baselines strictly on TRAIN split
    train_baselines = compute_train_baselines(train_raw)
    print("  TRAIN baselines computed:", list(train_baselines["P"].keys()))

    train_raw = apply_power_and_dynamic_features(train_raw, train_baselines)
    val_raw = apply_power_and_dynamic_features(val_raw, train_baselines)
    test_raw = apply_power_and_dynamic_features(test_raw, train_baselines)

    print("\n" + "=" * 70)
    print("F. REMAP")
    print("=" * 70)
    if REMAP_LABELS:
        train_raw["label"] = train_raw["label"].replace(REMAP_LABELS)
        val_raw["label"] = val_raw["label"].replace(REMAP_LABELS)
        test_raw["label"] = test_raw["label"].replace(REMAP_LABELS)

    print("\n" + "=" * 70)
    print("G. CLEAN")
    print("=" * 70)
    train = clean_dataframe(train_raw, "TRAIN")
    val = clean_dataframe(val_raw, "VALIDATION")
    test = clean_dataframe(test_raw, "TEST")

    print("\n" + "=" * 70)
    print("H. VALIDATE")
    print("=" * 70)
    train = validate_dataframe(train, "TRAIN")
    val = validate_dataframe(val, "VALIDATION")
    test = validate_dataframe(test, "TEST")

    print("\n" + "=" * 70)
    print("I. FEATURE SELECTION")
    print("=" * 70)
    train = select_features(train, "TRAIN")
    val = select_features(val, "VALIDATION")
    test = select_features(test, "TEST")

    print("\n" + "=" * 70)
    print("J. WINDOWING")
    print("=" * 70)
    train_windows = make_windows(train, "TRAIN")
    val_windows = make_windows(val, "VALIDATION")
    test_windows = make_windows(test, "TEST")

    print("\n" + "=" * 70)
    print("K. REMOVE UNSEEN CLASSES")
    print("=" * 70)
    train_labels = set(train_windows["y"].tolist())
    val_windows = remove_unseen_windows(val_windows, train_labels, "VALIDATION")
    test_windows = remove_unseen_windows(test_windows, train_labels, "TEST")

    print("\n" + "=" * 70)
    print("L. ENCODING")
    print("=" * 70)
    ohe, label_to_idx = fit_encoding(train_windows)
    train_encoded = encode_windows(train_windows, ohe, label_to_idx, "TRAIN")
    val_encoded = encode_windows(val_windows, ohe, label_to_idx, "VALIDATION")
    test_encoded = encode_windows(test_windows, ohe, label_to_idx, "TEST")

    print("\n" + "=" * 70)
    print("M. SCALING")
    print("=" * 70)
    scaler = fit_scaler(train_encoded)
    train_final = apply_scaler(train_encoded, scaler)
    val_final = apply_scaler(val_encoded, scaler)
    test_final = apply_scaler(test_encoded, scaler)

    print("\n" + "=" * 70)
    print("N. SAVE")
    print("=" * 70)
    save_npz(train_final, OUTPUT_DIR / "train_final.npz")
    save_npz(val_final, OUTPUT_DIR / "val_final.npz")
    save_npz(test_final, OUTPUT_DIR / "test_final.npz")

    joblib.dump(ohe, OUTPUT_DIR / "substation_encoder.joblib")
    joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")
    with open(OUTPUT_DIR / "label_vocab.json", "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("O. FINAL CHECK")
    print("=" * 70)
    print("TRAIN:", train_final["X"].shape)
    print("VAL  :", val_final["X"].shape)
    print("TEST :", test_final["X"].shape)
    print("\nFeatures:")
    for i, feature in enumerate(NUMERIC_FEATURES):
        print(f"  {i:2d}: {feature}")
    print("\nLabel vocabulary:")
    print(label_to_idx)
    print("\nTRAIN labels:")
    values, counts = np.unique(train_windows["y"], return_counts=True)
    print(dict(zip(values, counts)))
    print("\nPipeline completed in " f"{time.time() - start_time:.1f}s")
    print("=" * 70)

if __name__ == "__main__":
    main()