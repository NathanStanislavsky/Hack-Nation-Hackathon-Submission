import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import your custom modules
from dataset import load_dataset
from model import BrainWaveIntentModel  # Ensure this class exists in model.py

torch.autograd.set_detect_anomaly(True)

# --- Configuration ---
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Label Mapping ---
# Maps the string labels from the dataset to integer IDs for CrossEntropyLoss
LABEL_MAP = {
    "Relax": 0,
    "Right Fist": 1,
    "Left Fist": 2,
    "Both Fists": 3,
    "Tongue Tapping": 4
}

def train():
    # 1. Load Dataset
    print(f"📂 Loading dataset...")
    full_dataset = load_dataset("./data")
    
    # Create DataLoader
    # num_workers=0 is usually safer for IterableDatasets with file IO
    dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE)

    # 2. Initialize Model
    print(f"🧠 Initializing model on {DEVICE}...")
    model = BrainWaveIntentModel().to(DEVICE)
    
    # 3. Setup Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    model.train()
    print("🚀 Starting training...")

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # --- Prepare Data ---
            # Move features to GPU/CPU
            eeg = batch['feature_eeg'].to(DEVICE).float()        # (B, 7499, 6)
            moments = batch['feature_moments'].to(DEVICE).float() # (B, 72, 40, 3, 2, 3)

            # --- Prepare Labels ---
            # The dataset returns a list of dictionaries for the label field
            # We need to extract the 'label' string and convert to int ID
            raw_labels = batch['label']['label'] # List of strings
            
            # Convert strings -> integers -> tensor
            target_ids = [LABEL_MAP[l] for l in raw_labels]
            targets = torch.tensor(target_ids).to(DEVICE)

            # --- Forward Pass ---
            optimizer.zero_grad()
            
            # Model is expected to return logits (B, 5)
            logits = model(eeg, moments)
            
            # --- Loss Calculation ---
            loss = criterion(logits, targets)
            
            # --- Backward Pass ---
            loss.backward()
            optimizer.step()

            # --- Metrics ---
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        # --- Epoch Summary ---
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = correct / total_samples
        print(f"✅ Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")

    print("🎉 Training complete.")

if __name__ == "__main__":
    train()