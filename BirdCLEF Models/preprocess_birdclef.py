import os
import math
import ast
import argparse
import logging
import shutil

import numpy as np
import librosa
from datasets import (
    load_dataset,
    Audio,
    Features,
    Value,
    Array3D,
    Sequence,
    logging as ds_logging
)
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s", level=level
    )
    # also reduce HF datasets logging if not verbose
    ds_logging.set_verbosity_info() if verbose else ds_logging.set_verbosity_warning()


def build_label_encoder(taxonomy_csv: str) -> Tuple[LabelEncoder, int]:
    logging.info("Loading taxonomy from %s", taxonomy_csv)
    taxonomy = load_dataset(
        "csv", data_files={"train": taxonomy_csv}, split="train"
    )
    labels = sorted(taxonomy.unique("primary_label"))
    le = LabelEncoder().fit(labels)
    num = len(le.classes_)
    logging.info("Found %d unique classes", num)
    return le, num


def add_audio_path(example, audio_dir: str):
    example["audio_filepath"] = os.path.join(audio_dir, example["filename"])
    return example


def extract_logmel_chunks_batched(batch, le: LabelEncoder, num_classes: int,
                                  sr: int, n_mels: int, hop_length: int,
                                  chunk_samples: int, n_frames: int):
    feats, labs, fnames = [], [], []
    for i in range(len(batch["primary_label"])):
        info = batch["audio_filepath"][i]
        y = info.get("array")
        if y is None or len(y) < sr * 0.5:
            continue
        vec = np.zeros(num_classes, dtype=np.float32)
        # primary label
        try:
            idx = le.transform([str(batch["primary_label"][i])])[0]
            vec[idx] = 1.0
        except Exception:
            pass
        # secondary labels
        sec = batch.get("secondary_labels", [""])[i] or "[]"
        try:
            sec_list = ast.literal_eval(sec)
            if not isinstance(sec_list, (list, tuple)):
                sec_list = [sec_list]
        except Exception:
            sec_list = []
        for code in sec_list:
            try:
                idx = le.transform([str(code)])[0]
                vec[idx] = 1.0
            except Exception:
                continue
        if vec.sum() == 0:
            continue
        # chunking
        total_chunks = math.ceil(len(y) / chunk_samples)
        for c in range(total_chunks):
            start = c * chunk_samples
            chunk = y[start: start + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode="constant")
            try:
                S = librosa.feature.melspectrogram(
                    y=chunk, sr=sr, n_mels=n_mels,
                    hop_length=hop_length, fmax=sr // 2
                )
                logS = librosa.power_to_db(S, ref=np.max)
                # pad or truncate time frames
                if logS.shape[1] < n_frames:
                    pad_width = n_frames - logS.shape[1]
                    logS = np.pad(logS, ((0, 0), (0, pad_width)), mode="constant", constant_values=logS.min())
                else:
                    logS = logS[:, :n_frames]
                if not np.isfinite(logS).all() or logS.max() == logS.min():
                    continue
                feats.append(logS[..., None].astype(np.float32))
                labs.append(vec)
                fnames.append(batch["filename"][i])
            except Exception as ex:
                logging.warning(
                    "Error processing chunk %d of %s: %s", c, batch["filename"][i], ex
                )
                continue
    return {"features": feats, "labels": labs, "source_filename": fnames}


def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess bird audio into a Hugging Face Datasets Arrow dataset"
    )
    p.add_argument("--root", type=str, required=True,
                   help="Base directory containing train.csv, taxonomy.csv, and train_audio/")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Where to save processed dataset")
    p.add_argument("--sample_rate", type=int, default=32000,
                   help="Resample rate for audio")
    p.add_argument("--n_mels", type=int, default=128,
                   help="Number of Mel bands")
    p.add_argument("--chunk_dur", type=float, default=5.0,
                   help="Chunk duration in seconds")
    p.add_argument("--hop_length", type=int, default=512,
                   help="Hop length for STFT")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Batch size for datasets.map()")
    p.add_argument("--num_proc", type=int, default=1,
                   help="Number of parallel processes (1 for sequential)")
    p.add_argument("--verbose", action="store_true",
                   help="Enable debug logging and more HF verbosity")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    TAXONOMY_CSV = os.path.join(args.root, "taxonomy.csv")
    TRAIN_CSV = os.path.join(args.root, "train.csv")
    AUDIO_DIR = os.path.join(args.root, "train_audio")
    SR = args.sample_rate
    N_MELS = args.n_mels
    CHUNK_SAMPLES = int(args.chunk_dur * SR)
    HOP_LENGTH = args.hop_length
    N_FRAMES = math.ceil(CHUNK_SAMPLES / HOP_LENGTH)

    # Build label encoder
    le, NUM_CLASSES = build_label_encoder(TAXONOMY_CSV)

    # Load metadata
    logging.info("Loading training metadata from %s", TRAIN_CSV)
    ds = load_dataset(
        "csv", data_files={"train": TRAIN_CSV}, split="train"
    )
    ds = ds.map(lambda ex: add_audio_path(ex, AUDIO_DIR), num_proc=1)
    ds = ds.cast_column("audio_filepath", Audio(sampling_rate=SR))

    # Define output schema
    output_feats = Features({
        "features": Array3D(dtype="float32", shape=(N_MELS, N_FRAMES, 1)),
        "labels": Sequence(Value("float32"), length=NUM_CLASSES),
        "source_filename": Value("string")
    })

    # Remove existing output
    if os.path.exists(args.output_dir):
        logging.info("Removing existing output directory %s", args.output_dir)
        shutil.rmtree(args.output_dir)

    # Map & process
    logging.info(
        "Starting feature extraction: batch_size=%d, num_proc=%d", args.batch_size, args.num_proc
    )
    ds_proc = ds.map(
        lambda batch: extract_logmel_chunks_batched(
            batch, le, NUM_CLASSES, SR, N_MELS,
            HOP_LENGTH, CHUNK_SAMPLES, N_FRAMES
        ),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=ds.column_names,
        features=output_feats,
    )
    logging.info("Processed dataset size: %d examples", len(ds_proc))

    # Save
    logging.info("Saving processed dataset to %s", args.output_dir)
    ds_proc.save_to_disk(args.output_dir)
    logging.info("Processing complete.")


if __name__ == "__main__":
    main()
