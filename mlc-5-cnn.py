import os
import numpy as np
import pandas as pd
import librosa
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from sklearn.preprocessing import LabelEncoder
from flax.training import train_state
from tqdm.auto import tqdm
import orbax.checkpoint as ocp
import ast
import math

# --- Configuration ---
ROOT = os.path.join(os.getcwd(), "birdclef-2025") 
TRAIN_CSV = os.path.join(ROOT, "train.csv")
TRAIN_AUDIO_DIR = os.path.join(ROOT, "train_audio")
TAXONOMY_CSV = os.path.join(ROOT, "taxonomy.csv") 
CHECKPOINT_DIR = "./bird_cnn_checkpoints"

SR = 32_000
N_MELS = 128
CHUNK_DURATION = 5
CHUNK_SAMPLES = CHUNK_DURATION * SR
HOP_LENGTH = 512
N_FRAMES_PER_CHUNK = math.ceil(CHUNK_SAMPLES / HOP_LENGTH)

BATCH_SIZE = 64
NUM_EPOCHS = 10
LR = 1e-4
SEED = 0

# --- Globals ---
le = None # LabelEncoder will be stored here
NUM_CLASSES = None # Number of classes will be stored here

def initialize_label_encoder_from_taxonomy(taxonomy_csv_path):
    """
    Initializes the global LabelEncoder (le) and NUM_CLASSES
    using the unique primary_label values from the taxonomy CSV file.
    """
    global le, NUM_CLASSES
    print(f"Initializing label encoder from: {taxonomy_csv_path}")
    try:
        taxonomy_df = pd.read_csv(taxonomy_csv_path)
    except FileNotFoundError:
        print(f"Error: Taxonomy CSV not found at {taxonomy_csv_path}")
        return False 

    if 'primary_label' not in taxonomy_df.columns:
        print(f"Error: 'primary_label' column not found in {taxonomy_csv_path}")
        return False

    unique_labels = sorted(list(taxonomy_df['primary_label'].astype(str).unique()))
    if not unique_labels:
         print(f"Error: No unique labels found in 'primary_label' column of {taxonomy_csv_path}")
         return False 

    # Create and fit the LabelEncoder
    le = LabelEncoder().fit(unique_labels)
    NUM_CLASSES = len(le.classes_)

    print(f"Label encoder initialized successfully: {NUM_CLASSES} classes found from taxonomy.")
    return True 


def prepare_chunked_data(df_train): # Takes the training dataframe as input
    """
    Prepares data by splitting audio into fixed-duration chunks,
    calculating log-mel spectrograms for each chunk, and assigning
    the file's labels (from df_train) to each chunk using the global 'le'.
    """
    global le, NUM_CLASSES, SR, N_MELS, CHUNK_SAMPLES, N_FRAMES_PER_CHUNK, HOP_LENGTH, TRAIN_AUDIO_DIR
    if le is None or NUM_CLASSES is None:
        raise ValueError("Global LabelEncoder (le) or NUM_CLASSES not initialized. Run initialize_label_encoder_from_taxonomy first.")

    print("Extracting features & labels (chunk-based) using training data labels...")

    features_list = []
    labels_list = []

    if 'secondary_labels' not in df_train.columns:
        df_train['secondary_labels'] = "[]"
    else:
        df_train['secondary_labels'] = df_train['secondary_labels'].fillna("[]")

    for row in tqdm(df_train.itertuples(index=False), total=len(df_train), desc="Processing Audio"):
        filename = row.filename
        primary_label_str = str(row.primary_label)
        secondary_labels_str = row.secondary_labels # Already filledna with '[]'

        try:
            label_vector = np.zeros(NUM_CLASSES, dtype=np.float32)
            try:
                label_idx = le.transform([primary_label_str])[0]
                label_vector[label_idx] = 1.0
            except ValueError:
                print(f"Warning: Primary label '{primary_label_str}' from {filename} not found in taxonomy. Skipping label.")
                pass
            try:
                sec_list = ast.literal_eval(secondary_labels_str)
                valid_secondary_indices = []
                codes_to_transform = []
                if isinstance(sec_list, (list, tuple)):
                    codes_to_transform = [c for c in sec_list if isinstance(c, str) and c]
                elif isinstance(sec_list, str) and sec_list: # Handle single string label
                    codes_to_transform = [sec_list]

                if codes_to_transform:
                    for code in codes_to_transform:
                        try:
                            valid_secondary_indices.append(le.transform([code])[0])
                        except ValueError:
                             print(f"Warning: Secondary label '{code}' from {filename} not found in taxonomy. Skipping label.")
                             pass 
                if valid_secondary_indices:
                    label_vector[valid_secondary_indices] = 1.0 
            except (ValueError, SyntaxError):
                 if isinstance(secondary_labels_str, str) and secondary_labels_str and secondary_labels_str != '[]' and secondary_labels_str != "''":
                     try: 
                         label_vector[le.transform([secondary_labels_str])[0]] = 1.0
                     except ValueError: pass 

            path = os.path.join(TRAIN_AUDIO_DIR, filename)
            y, _ = librosa.load(path, sr=SR, res_type='kaiser_fast')

            if len(y) < SR * 0.5: 
                 continue

            num_samples = len(y)
            num_chunks = math.ceil(num_samples / CHUNK_SAMPLES)

            for i in range(num_chunks):
                start_sample = i * CHUNK_SAMPLES
                end_sample = start_sample + CHUNK_SAMPLES
                chunk_audio = y[start_sample:end_sample]
                if len(chunk_audio) < CHUNK_SAMPLES:
                    padding = CHUNK_SAMPLES - len(chunk_audio)
                    chunk_audio = np.pad(chunk_audio, (0, padding), 'constant')

                S = librosa.feature.melspectrogram(
                    y=chunk_audio, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH, fmax=SR//2
                )
                log_S = librosa.power_to_db(S, ref=np.max)

                current_frames = log_S.shape[1]
                if current_frames < N_FRAMES_PER_CHUNK:
                    pad_width = N_FRAMES_PER_CHUNK - current_frames
                    log_S = np.pad(log_S, ((0, 0), (0, pad_width)), mode='minimum')
                elif current_frames > N_FRAMES_PER_CHUNK:
                    log_S = log_S[:, :N_FRAMES_PER_CHUNK]

                if log_S.size == 0 or not np.isfinite(log_S).all() or np.max(log_S) == np.min(log_S):
                    continue

                log_S_reshaped = log_S[..., np.newaxis].astype(np.float32)
                features_list.append(log_S_reshaped)
                labels_list.append(label_vector)

        except FileNotFoundError:
            print(f"Warning: Audio file not found {path}. Skipping.")
        except Exception as e:
            print(f"Error processing {filename} (chunking): {e}")
            pass

    if not features_list or not labels_list:
        print("Error: No valid features/labels were processed.")
        return None, None

    print(f"Processed {len(features_list)} total 5-second chunks from {len(df_train)} files.")
    features_np = np.stack(features_list)
    labels_np = np.stack(labels_list)
    X = jnp.array(features_np)
    Y = jnp.array(labels_np)
    print(f"Chunked dataset ready: X shape={X.shape}, Y shape={Y.shape}")
    return X, Y



class BirdCNN(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)) # Downsample
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)) # Downsample
        x = x.reshape((x.shape[0], -1)) # Flatten
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x


def create_train_state(rng, learning_rate):
    global NUM_CLASSES, N_MELS, N_FRAMES_PER_CHUNK
    if NUM_CLASSES is None: raise ValueError("NUM_CLASSES not set.")
    model = BirdCNN(num_classes=NUM_CLASSES)
    init_shape = (1, N_MELS, N_FRAMES_PER_CHUNK, 1)
    dummy_input = jnp.ones(init_shape, dtype=jnp.float32)
    params = model.init(rng, dummy_input)["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch_x, batch_y):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch_x)
        loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, batch_y))
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss_val, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_val

# --- Training Loop ---
def main_train(X, Y):
    print("Starting training...")
    rng = jax.random.PRNGKey(SEED)
    rng, init_rng = jax.random.split(rng)
    num_train = X.shape[0] 

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    mngr_options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    mngr = ocp.CheckpointManager(CHECKPOINT_DIR, options=mngr_options)

    state = create_train_state(init_rng, LR)
    start_epoch = 0

    latest_step = mngr.latest_step()
    if latest_step is not None:
        try:
            print(f"Attempting to restore checkpoint from step {latest_step}...")
            state = mngr.restore(latest_step, args=ocp.args.StandardRestore(state))
            start_epoch = latest_step + 1
            print(f"Restored checkpoint. Resuming from epoch {start_epoch + 1}")
        except Exception as e:
             print(f"Could not restore checkpoint {latest_step}: {e}. Starting fresh.")

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_num = epoch + 1
        perm_key, rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_key, num_train)
        epoch_loss = 0.0
        num_batches = 0

        with tqdm(range(0, num_train, BATCH_SIZE), desc=f"Epoch {epoch_num}/{NUM_EPOCHS}", leave=False) as pbar:
            for i in pbar:
                batch_indices = perm[i : i + BATCH_SIZE]
                if len(batch_indices) == 0: continue
                batch_x = X[batch_indices]
                batch_y = Y[batch_indices]
                state, loss_val = train_step(state, batch_x, batch_y)
                epoch_loss += loss_val
                num_batches += 1
                pbar.set_postfix(loss=f"{epoch_loss/num_batches:.4f}")

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch_num} done. Avg Loss: {avg_epoch_loss:.4f}")

        try:
            mngr.save(epoch, args=ocp.args.StandardSave(state))
            mngr.wait_until_finished()
            print(f"Checkpoint saved for epoch {epoch_num} (step {epoch})")
        except Exception as e:
            print(f"Checkpoint failed for epoch {epoch_num}: {e}")

    mngr.close()
    print("Training finished.")
    return state

if __name__ == "__main__":

    init_success = initialize_label_encoder_from_taxonomy(TAXONOMY_CSV)
    if not init_success:
        print("Exiting due to label encoder initialization failure.")
        exit()

    print(f"Loading training data CSV from: {TRAIN_CSV}")
    try:
        df_train = pd.read_csv(TRAIN_CSV)
    except FileNotFoundError:
        print(f"Error: Training CSV not found at {TRAIN_CSV}. Exiting.")
        exit()
    except Exception as e:
        print(f"Error reading training CSV {TRAIN_CSV}: {e}. Exiting.")
        exit()

    X_train, Y_train = prepare_chunked_data(df_train)
    if X_train is None or Y_train is None:
        print("Exiting due to data preparation failure.")
        exit()

    if X_train.shape[0] == 0:
        print("Error: No training data (chunks) were created. Exiting.")
        exit()

    final_state = main_train(X_train, Y_train)

    if final_state:
        print("Training complete. Final model state available.")
    else:
        print("Training did not complete successfully.")

    print("Script finished.")