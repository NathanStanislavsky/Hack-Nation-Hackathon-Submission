import matplotlib.pyplot as plt
import torch
import os
from tqdm import tqdm

from dataset import load_dataset

def save_individual_histograms(dataset, base_dir="./histograms", num_samples=100):
    """
    Saves a separate histogram for every single channel to verify scaling.
    """
    eeg_dir = os.path.join(base_dir, "eeg_channels")
    moments_dir = os.path.join(base_dir, "moment_features")
    os.makedirs(eeg_dir, exist_ok=True)
    os.makedirs(moments_dir, exist_ok=True)
    
    all_eeg = []
    all_moments = []

    print(f"⌛ Collecting data from {num_samples} samples...")
    count = 0
    for batch in dataset:
        all_eeg.append(batch["eeg_features"])
        all_moments.append(batch["moment_features"])
        count += 1
        if count >= num_samples:
            break

    # Shape: [Total_Time_Steps, 6]
    eeg_data = torch.cat(all_eeg, dim=0) 
    # Shape: [Total_Samples, 720]
    moment_data = torch.cat(all_moments, dim=0)

    # --- 1. Save EEG Plots ---
    print(f"📊 Generating 6 EEG histograms...")
    for i in range(6):
        plt.figure(figsize=(6, 4))
        channel_data = eeg_data[:, i].numpy().flatten()
        
        plt.hist(channel_data, bins=200, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f"EEG Channel {i} Normalization")
        plt.xlabel("Scaled Value")
        plt.ylabel("Frequency")
        plt.xlim(-0.1, 1.1)
        plt.grid(axis='y', alpha=0.3)
        
        plt.savefig(os.path.join(eeg_dir, f"eeg_ch_{i}.png"))
        plt.close()

    # --- 2. Save Moment Plots ---
    print(f"📊 Generating Moment histograms...")
    for i in tqdm(range(moment_data.size(1))):
        feat_data = moment_data[:, i].numpy().flatten()
        
        # Optional: Skip plotting if the channel is entirely empty/constant
        if feat_data.max() == feat_data.min():
            continue

        plt.figure(figsize=(6, 4))
        plt.hist(feat_data, bins=50, color='salmon', edgecolor='black', alpha=0.7)
        plt.title(f"Moment Feature {i} Normalization")
        plt.xlabel("Scaled Value")
        plt.ylabel("Frequency")
        plt.xlim(-0.1, 1.1)
        plt.grid(axis='y', alpha=0.3)
        
        plt.savefig(os.path.join(moments_dir, f"moment_feat_{i:03d}.png"))
        plt.close()

    print(f"✅ Done! Plots saved to:\n - {eeg_dir}\n - {moments_dir}")

if __name__ == "__main__":
    # Ensure your load_dataset function is defined above this in your script
    ds = load_dataset()
    save_individual_histograms(ds, num_samples=50)