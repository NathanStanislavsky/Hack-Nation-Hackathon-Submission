import os
import shutil
import glob
import json
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset
from huggingface_hub import snapshot_download
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# --- Configuration ---
REPO_ID = "KernelCo/robot_control"
REPO_TYPE = "dataset"
ALLOW_PATTERNS = "data/*.npz"
SCALING_CONFIG_PATH = "scaling_config.json"
RAW_DIR = "./data/raw"
PROCESSED_DIR = "./data/processed"

LOWER_PCT = 1.0
UPPER_PCT = 99.0
NUM_SAMPLES_FOR_STATS = 1000

LABEL_NAMES = [
    "Right Fist",
    "Left Fist",
    "Both Fists",
    "Tongue Tapping",
    "Relax",
]

# --- Helper Functions (Must be at top level for pickling) ---

def _normalize_tensor(features, ranges):
    """Helper to apply min-max scaling based on pre-computed ranges."""
    # Convert ranges to tensor for vectorized ops (1, Channels) or (Channels,)
    lows = torch.tensor([r[0] for r in ranges], dtype=torch.float32)
    highs = torch.tensor([r[1] for r in ranges], dtype=torch.float32)
    
    if features.dim() == 2: # EEG (Time x Channels)
        lows = lows.unsqueeze(0)
        highs = highs.unsqueeze(0)
        
    features = torch.clamp(features, lows, highs)
    return (features - lows) / (highs - lows)

def _load_raw_file_for_stats(file_path):
    """Worker function to load a single file for statistics calculation."""
    try:
        with np.load(file_path, allow_pickle=True) as data:
            eeg = torch.nan_to_num(torch.tensor(data["feature_eeg"]), nan=0.0)
            moments = torch.nan_to_num(torch.tensor(data["feature_moments"]), nan=0.0).view(-1)
            return eeg, moments
    except Exception:
        return None

def _process_and_save_worker(file_path, output_dir, eeg_ranges, moment_ranges):
    """Worker function to process and save a single file."""
    # OPTIMIZATION: Prevent PyTorch from spawning threads inside this process
    # This prevents CPU thrashing when running many processes.
    torch.set_num_threads(1)
    
    fname = os.path.basename(file_path).replace('.npz', '.pt')
    save_path = os.path.join(output_dir, fname)
    
    # Skip if already exists (resume capability)
    if os.path.exists(save_path):
        return

    try:
        with np.load(file_path, allow_pickle=True) as data:
            # 1. Load & NaN handle
            eeg = torch.nan_to_num(torch.tensor(data["feature_eeg"], dtype=torch.float32), nan=0.0)
            moments = torch.nan_to_num(torch.tensor(data["feature_moments"], dtype=torch.float32).reshape(72, -1), nan=0.0)
            
            # 2. Normalize
            eeg = _normalize_tensor(eeg, eeg_ranges)
            
            # Flatten moments for normalization, then reshape back
            moment_flat = moments.view(-1)
            moment_flat = _normalize_tensor(moment_flat, moment_ranges)
            moments = moment_flat.view(72, -1)

            # 3. Label
            label_data = data["label"]
            label_str = label_data.item() if isinstance(label_data, np.ndarray) else label_data
            label = torch.tensor(LABEL_NAMES.index(label_str["label"]))

            # 4. Save
            torch.save({
                "eeg_features": eeg.clone(),
                "moment_features": moments.clone(),
                "label": label
            }, save_path)
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

# --- Main Logic ---

class BrainWaveIntentDataset(IterableDataset):
    def __init__(self, data_dir, shuffle=True):
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        
        if not self.files:
            raise RuntimeError(f"No .pt files found in {data_dir}. Run prepare_data() first.")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        file_list = self.files.copy()
        
        if self.shuffle:
            random.shuffle(file_list)

        if worker_info is not None:
            per_worker = int(np.ceil(len(file_list) / float(worker_info.num_workers)))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(file_list))
            file_list = file_list[iter_start:iter_end]

        for file_path in file_list:
            try:
                yield torch.load(file_path)
            except Exception:
                continue

def _get_scaling_stats(raw_files):
    if os.path.exists(SCALING_CONFIG_PATH):
        print(f"📖 Loading scaling config from {SCALING_CONFIG_PATH}")
        with open(SCALING_CONFIG_PATH, "r") as f:
            config = json.load(f)
            if config.get("percentiles") == [LOWER_PCT, UPPER_PCT]:
                return config["EEG_RANGES"], config["MOMENT_RANGES"]

    print(f"🧪 Calculating stats on {NUM_SAMPLES_FOR_STATS} samples (Parallel)...")
    
    subset = raw_files[:NUM_SAMPLES_FOR_STATS]
    all_eeg, all_moments = [], []

    # Parallel Load for Stats
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(_load_raw_file_for_stats, subset), total=len(subset), desc="Loading Stats Data"))
    
    for res in results:
        if res:
            all_eeg.append(res[0])
            all_moments.append(res[1])

    print("📊 Computing percentiles...")
    eeg_tensor = torch.cat(all_eeg, dim=0)
    moments_tensor = torch.stack(all_moments, dim=0)

    eeg_ranges = []
    for i in range(eeg_tensor.shape[1]):
        low, high = np.percentile(eeg_tensor[:, i].numpy(), [LOWER_PCT, UPPER_PCT])
        eeg_ranges.append((float(low), float(high)))

    moment_ranges = []
    for i in range(moments_tensor.shape[1]):
        low, high = np.percentile(moments_tensor[:, i].numpy(), [LOWER_PCT, UPPER_PCT])
        if low == high: high += 1e-6
        moment_ranges.append((float(low), float(high)))

    with open(SCALING_CONFIG_PATH, "w") as f:
        json.dump({
            "percentiles": [LOWER_PCT, UPPER_PCT],
            "EEG_RANGES": eeg_ranges,
            "MOMENT_RANGES": moment_ranges
        }, f, indent=4)
    
    return eeg_ranges, moment_ranges

def prepare_data(raw_dir, processed_dir):
    # 1. Download
    if not os.path.exists(raw_dir) or not glob.glob(os.path.join(raw_dir, "*.npz")):
        print(f"⬇️  Downloading raw data...")
        snapshot_download(repo_id=REPO_ID, repo_type=REPO_TYPE, local_dir=raw_dir, allow_patterns=ALLOW_PATTERNS)
        # Flatten logic...
        nested_dir = os.path.join(raw_dir, "data")
        if os.path.exists(nested_dir):
            for filename in os.listdir(nested_dir):
                shutil.move(os.path.join(nested_dir, filename), os.path.join(raw_dir, filename))
            os.rmdir(nested_dir)

    raw_files = glob.glob(os.path.join(raw_dir, "*.npz"))
    os.makedirs(processed_dir, exist_ok=True)
    
    # 2. Stats
    eeg_ranges, moment_ranges = _get_scaling_stats(raw_files)

    # 3. Parallel Processing
    # Check what's left to do
    processed_files = set(os.path.basename(f).replace('.pt', '') for f in glob.glob(os.path.join(processed_dir, "*.pt")))
    files_to_process = [f for f in raw_files if os.path.basename(f).replace('.npz', '') not in processed_files]

    if not files_to_process:
        print("✅ All files already processed.")
        return

    print(f"🚀 Parallel Processing {len(files_to_process)} files using {os.cpu_count()} cores...")
    
    # Create a partial function to freeze the constant arguments
    worker_func = partial(
        _process_and_save_worker, 
        output_dir=processed_dir, 
        eeg_ranges=eeg_ranges, 
        moment_ranges=moment_ranges
    )

    # Execute in parallel
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(worker_func, files_to_process), total=len(files_to_process), desc="Preprocessing"))

def load_dataset(shuffle=True):
    prepare_data(RAW_DIR, PROCESSED_DIR)
    return BrainWaveIntentDataset(PROCESSED_DIR, shuffle=shuffle)

if __name__ == "__main__":
    # This block is required for multiprocessing on Windows/macOS
    ds = load_dataset()
    print("Dataset ready.")