# preprocess_with_hf_datasets.py

import os
import math
import ast
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import librosa

from datasets import load_dataset, Audio, DatasetDict, Features, Value, Array3D

# --- Configuration ---
ROOT = os.path.join(os.getcwd(), "birdclef-2025")
TRAIN_CSV = os.path.join(ROOT, "train.csv")
TAXONOMY_CSV = os.path.join(ROOT, "taxonomy.csv")
AUDIO_DIR = os.path.join(ROOT, "train_audio")
OUTPUT_NPZ = os.path.join(ROOT, "precomputed_features.npz")

SR = 32_000
N_MELS = 128
CHUNK_DURATION = 5
CHUNK_SAMPLES = CHUNK_DURATION * SR
HOP_LENGTH = 512
N_FRAMES = math.ceil(CHUNK_SAMPLES / HOP_LENGTH)

NUM_PROC = max(1, os.cpu_count() - 2)   # number of parallel map workers

# --- 1) Build LabelEncoder from taxonomy.csv ---
taxonomy = load_dataset(
    "csv", 
    data_files=TAXONOMY_CSV, 
    split="train"
)
labels = sorted(taxonomy.unique("primary_label"))
le = LabelEncoder().fit(labels)
NUM_CLASSES = len(le.classes_)

# --- 2) Load train metadata as a Dataset, cast the audio column ---
ds = load_dataset(
    "csv", 
    data_files=TRAIN_CSV,
    split="train",
    column_names=None,       # infer from header
)

# assume your CSV has columns: filename, primary_label, secondary_labels
# create a full file path column, and cast it to Audio so HF will load it for you
def add_audio_path(example):
    example["audio_filepath"] = os.path.join(AUDIO_DIR, example["filename"])
    return example

ds = ds.map(add_audio_path)

ds = ds.cast_column("audio_filepath", Audio(sampling_rate=SR))

# --- 3) Define your feature‑extraction + chunking function ---
def extract_logmel_chunks(batch):
    """
    Input batch contains one example with keys:
      - batch["audio_filepath"]["array"] (loaded by HF Audio)
      - batch["primary_label"], batch["secondary_labels"]
    We return N chunked spectrograms (+ label vectors) as lists;
    HuggingFace will flatten them into multiple examples.
    """
    y = batch["audio_filepath"]["array"]
    if y is None or len(y) < SR * 0.5:
        return {"features": [], "labels": []}
    
    # build label vector
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    # primary
    try:
        vec[ le.transform([str(batch["primary_label"])])[0] ] = 1.0
    except ValueError:
        pass
    # secondary
    sec = batch.get("secondary_labels") or "[]"
    try:
        sec_list = ast.literal_eval(sec)
    except Exception:
        sec_list = []
    for code in sec_list if isinstance(sec_list, (list,tuple)) else [sec_list]:
        try:
            idx = le.transform([str(code)])[0]
            vec[idx] = 1.0
        except ValueError:
            continue
    if vec.sum() == 0:
        return {"features": [], "labels": []}
    
    # chunk & compute log-mel
    num_chunks = math.ceil(len(y) / CHUNK_SAMPLES)
    feats, labs = [], []
    for i in range(num_chunks):
        start = i * CHUNK_SAMPLES
        chunk = y[start : start + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)), mode="constant")
        S = librosa.feature.melspectrogram(
            y=chunk, sr=SR, n_mels=N_MELS,
            hop_length=HOP_LENGTH, fmax=SR//2
        )
        logS = librosa.power_to_db(S, ref=np.max)
        # pad/truncate to fixed width
        if logS.shape[1] < N_FRAMES:
            pad = N_FRAMES - logS.shape[1]
            logS = np.pad(logS, ((0,0),(0,pad)), mode="minimum")
        else:
            logS = logS[:,:N_FRAMES]
        # sanity check
        if not np.isfinite(logS).all() or logS.max() == logS.min():
            continue
        feats.append(logS[...,None].astype(np.float32))
        labs.append(vec)
    
    return {"features": feats, "labels": labs}

# --- 4) Map with batching/flattening, in parallel ---
# We tell HF that mapping will expand each input into multiple outputs
ds_feats = ds.map(
    extract_logmel_chunks,
    remove_columns=ds.column_names,
    batched=False,         # one input → many outputs
    num_proc=NUM_PROC,
    new_fingerprint="with-chunking",  # force cache invalidation if you change logic
    features=Features({
        "features": Array3D(dtype="float32", shape=(N_MELS, N_FRAMES, 1)),
        "labels":   Array3D(dtype="float32", shape=(NUM_CLASSES,)),  # HF will broadcast shape
    })
)

# --- 5) Convert to NumPy & save ---
X = np.stack(ds_feats["features"], axis=0)
Y = np.stack(ds_feats["labels"],   axis=0)
print(f"Saving {X.shape[0]} chunks as {OUTPUT_NPZ}")
np.savez_compressed(OUTPUT_NPZ, x=X, y=Y)