import os
import glob
import math
import copy
import random
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import load_dataset, PROCESSED_DIR
from model import BrainWaveIntentModel
from checkpoint import save_checkpoint, load_checkpoint

torch.set_float32_matmul_precision('high')

CONFIG = {
    "lr": 0,
    "batch_size": 64,
    "num_epochs": 100,
    "num_layers": 4,
    "dim": 1024,
    "save_every": 500,
    "val_split": 0.1,
    "checkpoint_path": "checkpoint.pt",
    "project_name": "brainwave-intent"
}

def get_dataset_size(data_dir):
    return len(glob.glob(os.path.join(data_dir, "*.pt")))

@torch.no_grad()
def validate(model, loader, device, criterion, use_amp):
    model.eval()
    total_loss = 0
    total_acc = 0
    steps = 0
    
    for batch in loader:
        eeg_features = batch["eeg_features"].to(device, non_blocking=True)
        moment_features = batch["moment_features"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16 if use_amp else torch.bfloat16):
            logits = model(eeg_features, moment_features)
            loss = criterion(logits, labels)

        acc = (logits.argmax(1) == labels).float().mean()
        total_loss += loss.item()
        total_acc += acc.item()
        steps += 1

    model.train()
    return total_loss / (steps + 1e-6), total_acc / (steps + 1e-6)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu"))
    
    # Data Preparation
    full_ds = load_dataset()
    all_files = full_ds.files
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * (1 - CONFIG["val_split"]))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    train_ds = full_ds
    train_ds.files = train_files
    
    val_ds = copy.copy(full_ds)
    val_ds.files = val_files

    steps_per_epoch = math.ceil(len(train_files) / CONFIG["batch_size"])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], num_workers=2, pin_memory=True)
    
    model = BrainWaveIntentModel(num_layers=CONFIG["num_layers"], dim=CONFIG["dim"]).to(device)
    
    if hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()
    
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    start_global_step, wandb_id = load_checkpoint(model, optimizer, CONFIG["checkpoint_path"])
    start_epoch = start_global_step // steps_per_epoch

    wandb.init(
        project=CONFIG["project_name"],
        id=wandb_id,
        resume="allow",
        config=CONFIG
    )

    model.train()
    current_step = 0 

    try:
        for epoch in range(start_epoch, CONFIG["num_epochs"]):
            pbar = tqdm(train_loader, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}", dynamic_ncols=True)
            
            for batch in pbar:
                current_step += 1
                
                if current_step <= start_global_step:
                    continue

                eeg_features = batch["eeg_features"].to(device, non_blocking=True)
                moment_features = batch["moment_features"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, dtype=torch.float16 if use_amp else torch.bfloat16):
                    logits = model(eeg_features, moment_features)
                    loss = criterion(logits, labels)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                acc = (logits.argmax(1) == labels).float().mean()
                
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.4f}")
                
                wandb.log({
                    "train_loss": loss.item(), 
                    "train_acc": acc.item(), 
                    "lr": optimizer.param_groups[0]['lr'],
                    "epoch": epoch + 1,
                    "global_step": current_step
                })

                if current_step % CONFIG["save_every"] == 0:
                    save_checkpoint(model, optimizer, current_step, wandb.run.id, CONFIG["checkpoint_path"])

            # Validation Loop
            val_loss, val_acc = validate(model, val_loader, device, criterion, use_amp)
            print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            wandb.log({"val_loss": val_loss, "val_acc": val_acc, "epoch": epoch + 1})

            save_checkpoint(model, optimizer, current_step, wandb.run.id, CONFIG["checkpoint_path"])

    except KeyboardInterrupt:
        save_checkpoint(model, optimizer, current_step, wandb.run.id, CONFIG["checkpoint_path"])
    finally:
        wandb.finish()

if __name__ == "__main__":
    train()