import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from dataset import load_dataset

# Create output directory
os.makedirs("histograms", exist_ok=True)

ds = load_dataset()
dl = DataLoader(ds, batch_size=1) # Note: Increasing batch_size will be faster

# Storage for collecting all values
# We use lists to collect batches, then concatenate at the end
eeg_all = []
moments_0_all = [] # Log Intensity
moments_1_all = [] # Mean Time of Flight
moments_2_all = [] # Variance

print("Collecting data from dataset...")

for i, batch in enumerate(dl):
    # --- Collect EEG ---
    # Flatten to 1D array
    eeg_batch = batch["feature_eeg"].flatten().cpu().numpy()
    eeg_all.append(eeg_batch)

    # --- Collect Moments (Split by type) ---
    # Shape is (Batch, Time, Mods, SDS, Wave, 3)
    # We slice the last dimension [..., i]
    moments = batch["feature_moments"]
    
    # Index 0: Intensity
    m0 = moments[..., 0].flatten().cpu().numpy()
    moments_0_all.append(m0)
    
    # Index 1: Mean ToF
    m1 = moments[..., 1].flatten().cpu().numpy()
    moments_1_all.append(m1)
    
    # Index 2: Variance
    m2 = moments[..., 2].flatten().cpu().numpy()
    moments_2_all.append(m2)
    
    if i % 10 == 0:
        print(f"Processed batch {i}...", end="\r")

print(f"\nData collection complete. Processing histograms...")

# Concatenate all batches into single large arrays
eeg_final = np.concatenate(eeg_all)
m0_final = np.concatenate(moments_0_all)
m1_final = np.concatenate(moments_1_all)
m2_final = np.concatenate(moments_2_all)

# Helper function to save plots
def plot_hist(data, filename, title, color='blue', zoom=False):
    plt.figure(figsize=(10, 6))
    
    # If zoom is requested, clip to 1st and 99th percentile to hide outliers
    if zoom:
        low, high = np.percentile(data, [1, 99])
        data = data[(data >= low) & (data <= high)]
        title += f" (Clipped: {low:.1f} to {high:.1f})"
    
    plt.hist(data, bins=100, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    plt.title(title)
    plt.xlabel("Feature Value")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join("histograms", filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

# --- 1. EEG Histograms ---
# Plot raw (to see artifacts)
plot_hist(eeg_final, "eeg_raw.png", "EEG Data (Full Range incl. Artifacts)", color='red')
# Plot zoomed (to see actual brain signal distribution)
plot_hist(eeg_final, "eeg_zoomed.png", "EEG Data (Zoomed / Signal Only)", color='green', zoom=True)

# --- 2. Moments Histograms ---
plot_hist(m0_final, "moments_0_intensity.png", "Moments: Log Intensity (Idx 0)", color='purple')
plot_hist(m1_final, "moments_1_mean_tof.png", "Moments: Mean ToF (Idx 1)", color='orange')
plot_hist(m2_final, "moments_2_variance.png", "Moments: Variance (Idx 2)", color='cyan')

print("Done.")