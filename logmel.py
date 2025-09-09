import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import ast
import math
import time
import argparse

# --- Configuration ---
DEFAULT_ROOT = os.path.join(os.getcwd(), "birdclef-2025")
DEFAULT_TRAIN_CSV = os.path.join(DEFAULT_ROOT, "train.csv")
DEFAULT_TRAIN_AUDIO_DIR = os.path.join(DEFAULT_ROOT, "train_audio")
DEFAULT_TAXONOMY_CSV = os.path.join(DEFAULT_ROOT, "taxonomy.csv")
DEFAULT_OUTPUT_NPZ = os.path.join(DEFAULT_ROOT, "precomputed_features.npz")

# Audio Processing Parameters
SR = 32_000
N_MELS = 128
CHUNK_DURATION = 5  # seconds
CHUNK_SAMPLES = CHUNK_DURATION * SR
HOP_LENGTH = 512
N_FRAMES_PER_CHUNK = math.ceil(CHUNK_SAMPLES / HOP_LENGTH)

# --- Globals ---
le = None
NUM_CLASSES = None

# --- Label Encoder Initialization ---
def initialize_label_encoder_from_taxonomy(taxonomy_csv_path):
    global le, NUM_CLASSES
    taxonomy_df = pd.read_csv(taxonomy_csv_path)
    if 'primary_label' not in taxonomy_df.columns:
        raise ValueError("'primary_label' column not found in taxonomy CSV")
    unique_labels = sorted(taxonomy_df['primary_label'].astype(str).unique())
    le = LabelEncoder()
    le.fit(unique_labels)
    NUM_CLASSES = len(le.classes_)

# --- Single File Processing ---
def process_single_file(row, audio_dir):
    filename = row['filename']
    primary_label = str(row['primary_label'])
    secondary_labels = row.get('secondary_labels', '[]')

    # Build label vector
    label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    # Primary label
    try:
        idx = le.transform([primary_label])[0]
        label_vec[idx] = 1.0
    except ValueError:
        pass
    # Secondary labels
    try:
        sec_list = ast.literal_eval(secondary_labels)
        if isinstance(sec_list, (list, tuple)):
            codes = [str(c) for c in sec_list]
        else:
            codes = [str(sec_list)]
        for code in codes:
            try:
                idx = le.transform([code])[0]
                label_vec[idx] = 1.0
            except ValueError:
                pass
    except (ValueError, SyntaxError, TypeError):
        pass

    if label_vec.sum() == 0:
        return [], []

    # Load audio
    path = os.path.join(audio_dir, filename)
    try:
        y, _ = librosa.load(path, sr=SR, res_type='kaiser_fast')
    except FileNotFoundError:
        return [], []

    if len(y) < SR * 0.5:
        return [], []

    features = []
    labels = []
    num_samples = len(y)
    num_chunks = math.ceil(num_samples / CHUNK_SAMPLES)

    for i in range(num_chunks):
        start = i * CHUNK_SAMPLES
        end = min(start + CHUNK_SAMPLES, num_samples)
        chunk = y[start:end]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)), 'constant')

        S = librosa.feature.melspectrogram(
            y=chunk, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH, fmax=SR//2
        )
        log_S = librosa.power_to_db(S, ref=np.max)
        frames = log_S.shape[1]

        if frames < N_FRAMES_PER_CHUNK:
            pad = N_FRAMES_PER_CHUNK - frames
            log_S = np.pad(log_S, ((0, 0), (0, pad)), mode='minimum')
        elif frames > N_FRAMES_PER_CHUNK:
            log_S = log_S[:, :N_FRAMES_PER_CHUNK]

        if not np.isfinite(log_S).all() or np.max(log_S) <= np.min(log_S):
            continue

        features.append(log_S[..., np.newaxis].astype(np.float32))
        labels.append(label_vec)

    return features, labels

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess audio to log-mel features.")
    parser.add_argument('--train_csv', type=str, default=DEFAULT_TRAIN_CSV)
    parser.add_argument('--taxonomy_csv', type=str, default=DEFAULT_TAXONOMY_CSV)
    parser.add_argument('--audio_dir', type=str, default=DEFAULT_TRAIN_AUDIO_DIR)
    parser.add_argument('--output_npz', type=str, default=DEFAULT_OUTPUT_NPZ)
    parser.add_argument('--sample', type=int, default=None)
    args = parser.parse_args()

    print("Initializing label encoder...")
    initialize_label_encoder_from_taxonomy(args.taxonomy_csv)

    print("Loading training metadata...")
    df = pd.read_csv(args.train_csv)
    if 'secondary_labels' not in df.columns:
        df['secondary_labels'] = '[]'
    df['secondary_labels'] = df['secondary_labels'].fillna('[]')

    if args.sample:
        df = df.sample(n=args.sample, random_state=42)

    all_feats = []
    all_labels = []

    print("Processing files sequentially...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        feats, labs = process_single_file(row, args.audio_dir)
        all_feats.extend(feats)
        all_labels.extend(labs)

    if not all_feats:
        print("No valid features extracted. Exiting.")
        exit(1)

    print("Stacking arrays...")
    X = np.stack(all_feats, axis=0)
    Y = np.stack(all_labels, axis=0)
    print(f"Features shape: {X.shape}, Labels shape: {Y.shape}")

    os.makedirs(os.path.dirname(args.output_npz), exist_ok=True)
    print(f"Saving to {args.output_npz}...")
    np.savez_compressed(args.output_npz, x=X, y=Y)
    print("Done.")
