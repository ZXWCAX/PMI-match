import os

# Must be set before fork to ensure child processes inherit this env var, preventing underlying memory fragmentation
os.environ["MALLOC_ARENA_MAX"] = "1"

import sys
import csv
import gc
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib

# Force headless backend to completely disable Tkinter dependency
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import CADVectorFieldDataset
from loss import distance_field_loss


def worker_init_fn(worker_id):
    """Prevents memory leaks caused by DataLoader multi-processing"""
    import ctypes
    import numpy as np
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt(ctypes.c_int(-8), ctypes.c_int(1))
    except Exception:
        pass
    np.random.seed(worker_id)


# ==================================================
# Helper: Calculate Threshold Accuracy (< 5px error)
# ==================================================
def calculate_accuracy(pred_offsets, distance_map, bbox_mask, threshold=5.0):
    B, _, H, W = pred_offsets.shape
    device = pred_offsets.device

    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    xx = xx.float().unsqueeze(0).expand(B, -1, -1)
    yy = yy.float().unsqueeze(0).expand(B, -1, -1)

    pred_x = xx + pred_offsets[:, 0, :, : ]
    pred_y = yy + pred_offsets[:, 1, :, : ]

    norm_x = 2.0 * pred_x / (W - 1) - 1.0
    norm_y = 2.0 * pred_y / (H - 1) - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1)

    sampled_dist = F.grid_sample(distance_map, grid, mode='bilinear', padding_mode='border', align_corners=True)

    valid_pixels = bbox_mask.sum() + 1e-6
    correct_pixels = ((sampled_dist < threshold) * bbox_mask).sum()

    return (correct_pixels / valid_pixels).item()


# ==================================================
# Helper: Plot and Save Dual-Subplot Curves
# ==================================================
def plot_curves(history, save_path="training_curves.png"):
    epochs = range(1, len(history['train_loss']) + 1)
    fig = plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    plt.title('Loss Curve (Distance Field)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_acc'], 'g-', label='Val Accuracy (< 5px)', linewidth=2)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    fig.clf()
    plt.close(fig)
    plt.close('all')


# ==================================================
# Main Training Pipeline
# ==================================================
def train_model():
    BASE_DIR = "../output_dataset_batch"
    BATCH_SIZE = 8
    EPOCHS = 100  # Total training epochs set to 100
    MAX_EPOCHS_PER_RUN = 2  # Max epochs to train per run before auto-restart
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG_FILE = "training_log.csv"
    CHECKPOINT_PATH = "checkpoint_latest.pth"

    train_dataset = CADVectorFieldDataset(os.path.join(BASE_DIR, "train.json"), BASE_DIR, target_size=1024)
    val_dataset = CADVectorFieldDataset(os.path.join(BASE_DIR, "val.json"), BASE_DIR, target_size=1024)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, drop_last=True, persistent_workers=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, drop_last=False, persistent_workers=True,
        worker_init_fn=worker_init_fn
    )

    # Instantiate DeepLabV3+ architecture
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=4,
        classes=2,
        activation=None
    ).to(DEVICE)

    if torch.cuda.device_count() > 1:
        print(f"🚀 Detected {torch.cuda.device_count()} GPUs. Multi-GPU training enabled with DataParallel!")
        model = torch.nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = GradScaler()

    start_epoch = 0
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    # Resume from checkpoint logic
    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 Checkpoint found at {CHECKPOINT_PATH}. Restoring training state...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        # Restore historical logs for continuous plotting
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, mode='r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 4:
                        history['train_loss'].append(float(row[1]))
                        history['val_loss'].append(float(row[2]))
                        history['val_acc'].append(float(row[3]))
        print(f"✅ Successfully restored! Resuming training from Epoch {start_epoch + 1}.")
    else:
        with open(LOG_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_Accuracy'])

    if start_epoch >= EPOCHS:
        print("🎉 Model has already completed all 100 training epochs!")
        sys.exit(0)

    print(f"🚀 Starting DeepLabV3+ training (1024 resolution) | Device: {DEVICE} | Batch Size: {BATCH_SIZE}")

    epochs_trained_this_run = 0

    for epoch in range(start_epoch, EPOCHS):
        # ---------------- Training Phase ----------------
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        for batch in pbar:
            inputs = batch["input"].to(DEVICE)
            dist_maps = batch["distance_map"].to(DEVICE)
            bbox_masks = batch["bbox_mask"].to(DEVICE)

            optimizer.zero_grad()

            with autocast():
                pred_offsets = model(inputs)
                loss = distance_field_loss(pred_offsets, dist_maps, bbox_masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            del batch, inputs, dist_maps, bbox_masks, pred_offsets, loss

        avg_train_loss = train_loss / len(train_loader)

        # ---------------- Validation Phase ----------------
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]"):
                inputs = batch["input"].to(DEVICE)
                dist_maps = batch["distance_map"].to(DEVICE)
                bbox_masks = batch["bbox_mask"].to(DEVICE)

                with autocast():
                    pred_offsets = model(inputs)
                    loss = distance_field_loss(pred_offsets, dist_maps, bbox_masks)
                    acc = calculate_accuracy(pred_offsets, dist_maps, bbox_masks, threshold=5.0)

                val_loss += loss.item()
                val_accuracy += acc

                del batch, inputs, dist_maps, bbox_masks, pred_offsets, loss

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_accuracy / len(val_loader)

        print(
            f"📊 Epoch {epoch + 1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc (<5px): {avg_val_acc:.2%}")

        # ---------------- Logging & Plotting ----------------
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)

        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, avg_val_acc])

        plot_curves(history, save_path="training_curves.png")

        # ---------------- Model & Checkpoint Saving ----------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), "best_vector_field_model_deeplabv3plus_1024.pth")
            print("💾 New best validation loss achieved! Saved the best DeepLabV3+ model weights.")

        # Save latest state at the end of each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': (model.module if hasattr(model, 'module') else model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, CHECKPOINT_PATH)

        gc.collect()
        epochs_trained_this_run += 1

        # Periodic restart logic to prevent memory leaks
        if epochs_trained_this_run >= MAX_EPOCHS_PER_RUN:
            print(f"⏳ Trained {MAX_EPOCHS_PER_RUN} epochs continuously. Exiting to release memory. Waiting for external script to restart...")
            sys.exit(0)


if __name__ == "__main__":
    train_model()















