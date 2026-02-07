import torch
import torch.nn as nn
import torch.nn.functional as F

class BrainWaveIntentModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # --- Branch 1: EEG Processing ---
        # Input: (Batch, 7499, 6)
        # Fix: Normalize input immediately. 
        self.eeg_norm = nn.LayerNorm([7499, 6]) 
        
        self.eeg_conv1 = nn.Conv1d(6, 16, kernel_size=64, stride=2)
        self.eeg_bn1 = nn.BatchNorm1d(16)
        self.eeg_pool1 = nn.MaxPool1d(kernel_size=4)
        
        self.eeg_conv2 = nn.Conv1d(16, 32, kernel_size=32, stride=2)
        self.eeg_bn2 = nn.BatchNorm1d(32)
        self.eeg_pool2 = nn.MaxPool1d(kernel_size=4)
        
        self.eeg_global_pool = nn.AdaptiveAvgPool1d(1) 

        # --- Branch 2: Moments Processing ---
        self.moments_flat_size = 72 * 40 * 3 * 2 * 3 
        
        # Fix: Normalize the massive flat vector
        self.moments_norm = nn.LayerNorm(self.moments_flat_size)

        self.moments_fc = nn.Sequential(
            nn.Linear(self.moments_flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # --- Fusion ---
        self.classifier = nn.Sequential(
            nn.Linear(32 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes) 
        )

    def forward(self, eeg, moments):
        # 1. Normalize Inputs
        x_eeg = self.eeg_norm(eeg)
        
        # Flatten moments first, then normalize
        x_moments = moments.view(moments.size(0), -1)
        x_moments = self.moments_norm(x_moments)

        # 2. Process EEG
        x_eeg = x_eeg.permute(0, 2, 1) # (Batch, Channels, Time)
        x_eeg = self.eeg_pool1(F.relu(self.eeg_bn1(self.eeg_conv1(x_eeg))))
        x_eeg = self.eeg_pool2(F.relu(self.eeg_bn2(self.eeg_conv2(x_eeg))))
        x_eeg = self.eeg_global_pool(x_eeg).squeeze(-1)

        # 3. Process Moments
        x_moments = self.moments_fc(x_moments)

        # 4. Combine
        combined = torch.cat((x_eeg, x_moments), dim=1)
        return self.classifier(combined)