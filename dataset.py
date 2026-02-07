import os
import shutil
import glob
import numpy as np
import torch
from torch.utils.data import IterableDataset
from huggingface_hub import snapshot_download

REPO_ID = "KernelCo/robot_control"
REPO_TYPE = "dataset"
ALLOW_PATTERNS = "data/*.npz"

class BrainWaveIntentDataset(IterableDataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.files:
            print(f"⚠️  Warning: No .npz files found in {data_dir}")

    def __iter__(self):
        for file_path in self.files:
            try:
                # allow_pickle=True is required for object arrays (strings, dicts)
                with np.load(file_path, allow_pickle=True) as data:
                    yield self._preprocess(data["feature_eeg"], data["feature_moments"], data["label"].item())

            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
    
    def _preprocess(self, feature_eeg, feature_moments, label):
        
        feature_eeg = torch.tensor(feature_eeg)
        feature_eeg = torch.nan_to_num(feature_eeg, nan=0.0)
        feature_eeg = torch.clamp(feature_eeg, -100000, 300000)

        feature_moments = torch.tensor(feature_moments)
        feature_moments[:, :, :, :, 1] = torch.clamp(feature_moments[:, :, :, :, 1], 1750, 3250)
        feature_moments[:, :, :, :, 2] = torch.clamp(feature_moments[:, :, :, :, 2], 75000, 300000)

        return {
            "feature_eeg": feature_eeg,
            "feature_moments": feature_moments,
            "label": label
        }
        

def _download_and_flatten(local_dir):
    print(f"⬇️  Downloading {REPO_ID} to {local_dir}...")
    
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=local_dir,
        allow_patterns=ALLOW_PATTERNS,
        resume_download=True
    )

    # Flatten logic
    nested_dir = os.path.join(local_dir, "data")
    if os.path.exists(nested_dir):
        for filename in os.listdir(nested_dir):
            shutil.move(os.path.join(nested_dir, filename), os.path.join(local_dir, filename))
        os.rmdir(nested_dir)

def load_dataset(local_dir="./data"):
    # Check if directory exists and has files
    if not (os.path.exists(local_dir) and len(glob.glob(os.path.join(local_dir, "*.npz"))) > 0):
        _download_and_flatten(local_dir)
    else:
        print(f"✅ BrainWaveIntentDataset found in '{local_dir}'. Skipping download.")

    return BrainWaveIntentDataset(local_dir)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    ds = load_dataset()
    
    for batch in ds:
        print("example batch:", batch)
        break