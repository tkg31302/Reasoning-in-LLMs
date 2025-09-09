import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm
import time

print("SIMULATION MODE: Will run in ~10 seconds")

# --- Minimal Configuration ---
ROOT = "/kaggle/input/birdclef-2025"
TRAIN_CSV = os.path.join(ROOT, "train.csv")
OUTPUT_SUB = "quick_test_submission.csv"

# ULTRA-MINIMAL PARAMETERS
MAX_SAMPLES = 10  # Just 10 samples!
FEATURE_SIZE = 16  # Tiny feature size

# --- Globals ---
le = None
NUM_CLASSES = None

# --- Label Initialization ---
def initialize_globals(csv_path):
    global le, NUM_CLASSES
    print(f"Fast-loading labels from: {csv_path}")
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"Limiting to {MAX_SAMPLES} samples for simulation")
            df = df.head(MAX_SAMPLES)
        else:
            print(f"CSV not found, using mock data")
            # Create mock data if file doesn't exist
            df = pd.DataFrame({
                'primary_label': ['species1', 'species2', 'species3'] * 4,
                'secondary_labels': ['[]'] * 10,
                'filename': [f'mock_file_{i}.ogg' for i in range(10)]
            })
        
        all_codes = set(df['primary_label'].astype(str))
        le = LabelEncoder().fit(list(all_codes))
        NUM_CLASSES = len(le.classes_)
        print(f"Label encoder initialized: {NUM_CLASSES} classes found")
        return df
        
    except Exception as e:
        print(f"Error: {e}")
        # Create emergency mock data
        le = LabelEncoder().fit(['species1', 'species2', 'species3'])
        NUM_CLASSES = 3
        df = pd.DataFrame({
            'primary_label': ['species1', 'species2', 'species3'] * 4,
            'secondary_labels': ['[]'] * 10,
            'filename': [f'mock_file_{i}.ogg' for i in range(10)]
        })
        return df

# --- Simulated Data Preparation ---
def prepare_data_simulated(df):
    global le, NUM_CLASSES
    if le is None or NUM_CLASSES is None:
        raise ValueError("Globals not initialized.")
    
    print("Generating simulated features (fast)...")
    start_time = time.time()
    
    # Generate random features instead of loading actual audio
    n_samples = len(df)
    features = np.random.randn(n_samples, FEATURE_SIZE).astype(np.float32)
    
    # Generate labels
    y_labels = le.transform(df['primary_label'])
    
    print(f"Data prepared in {time.time() - start_time:.2f}s")
    print(f"Dataset ready: X shape={features.shape}, y shape={y_labels.shape}")
    return features, y_labels

# --- Quick Training Model ---
def fast_train(X, y):
    print("Running minimal training...")
    start_time = time.time()
    
    # Use simple LogisticRegression instead of neural networks
    model = LogisticRegression(max_iter=100, multi_class='ovr')
    model.fit(X, y)
    
    # Report simple metrics
    train_preds = model.predict(X)
    accuracy = (train_preds == y).mean()
    print(f"Training accuracy: {accuracy:.2f}")
    
    print(f"Training completed in {time.time() - start_time:.2f}s")
    return model

# --- Mock Submission Generation ---
def generate_quick_submission(model):
    print("Generating simulated submission...")
    start_time = time.time()
    
    # Create dummy submission
    n_rows = 10
    
    # Generate random test features
    test_features = np.random.randn(n_rows, FEATURE_SIZE)
    
    # Get predictions
    probs = model.predict_proba(test_features)
    
    # Create sample submission dataframe
    row_ids = [f"testfile_{i}_5" for i in range(n_rows)]
    
    # Create DataFrame with all class probabilities
    result_df = pd.DataFrame(columns=["row_id"] + list(le.classes_))
    result_df["row_id"] = row_ids
    
    # Fill in the probabilities
    for i, class_idx in enumerate(model.classes_):
        class_name = le.inverse_transform([class_idx])[0]
        result_df[class_name] = probs[:, i]
    
    # Save submission
    result_df.to_csv(OUTPUT_SUB, index=False)
    print(f"Submission saved to {OUTPUT_SUB} in {time.time() - start_time:.2f}s")

# --- Main Execution ---
if __name__ == "__main__":
    overall_start = time.time()
    print("🚀 Starting ultra-fast simulation mode")
    
    # 1. Initialize globals with minimal samples
    df_train = initialize_globals(TRAIN_CSV)
    
    # 2. Generate simulated features
    X_train, y_train = prepare_data_simulated(df_train)
    
    # 3. Quick training
    final_model = fast_train(X_train, y_train)
    
    # 4. Generate mock submission
    generate_quick_submission(final_model)
    
    total_time = time.time() - overall_start
    print(f"✅ Completed in {total_time:.2f} seconds")
