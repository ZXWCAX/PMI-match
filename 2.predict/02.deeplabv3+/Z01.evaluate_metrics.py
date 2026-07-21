import os
import sys
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree

try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("❌ Missing segmentation_models_pytorch library!")
    sys.exit(1)

try:
    from loss import MappingLoss
except ImportError:
    print("❌ Cannot find loss.py. Please ensure it is in the same directory as this script!")
    sys.exit(1)

# ==================================================
# 1. Configuration & Path Specifications
# ==================================================
MODEL_PATH = "best_vector_field_model_deeplabv3plus_1024.pth"
VAL_JSON_PATH = "../output_dataset_batch/val.json"
BASE_DATASET_DIR = "../output_dataset_batch"

SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRIORITY_MAP = {"Highest": 4, "High": 3, "Medium": 2, "Low": 1}


# ==================================================
# 2. Helper Functions
# ==================================================
def cv2_imread_unicode(filepath, flags=cv2.IMREAD_COLOR):
    """Reads image files containing Unicode characters in paths."""
    if not os.path.exists(filepath):
        return None
    return cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), flags)


def get_highest_priority_faceid(face_refs):
    """Extracts the FaceID with the highest priority from faceRefs."""
    if not face_refs:
        return 0
    sorted_refs = sorted(face_refs, key=lambda x: PRIORITY_MAP.get(x.get("priority", "Low"), 0), reverse=True)
    return sorted_refs[0]["faceID"]


# ==================================================
# 3. Main Evaluation Pipeline
# ==================================================
def main():
    print("🚀 Loading DeepLabV3+ model and MappingLoss...")

    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=4,
        classes=2,
        activation=None
    )

    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device)
        new_state_dict = {k.replace('module.', '').replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print("✅ Weights loaded successfully!")
    else:
        print(f"❌ Weight file not found: {MODEL_PATH}")
        return

    model.to(device)
    model.eval()

    # Initialize loss function
    criterion = MappingLoss(lambda_reg=1.0, lambda_cos=0.5, lambda_tv=0.1).to(device)

    with open(VAL_JSON_PATH, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    # --- Statistics Accumulators ---
    total_mask_pts = 0
    total_exact_hits = 0
    total_5px_hits = 0
    total_10px_hits = 0
    total_distance_sum = 0.0  # Force float64 accumulation
    total_cos_sim_sum = 0.0

    total_loss = 0.0
    valid_batches = 0

    print(f"\n🎯 Starting deep quantitative evaluation (Total samples: {len(val_data)})...")
    pbar = tqdm(val_data, desc="Evaluation Progress", unit="sample", colour="cyan")

    temp_edge_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)

    for item in pbar:
        item_id = item['id']
        parts = item_id.split('_')
        gt_faceid = int(parts[-2]) if parts[-1] == 'aug' else int(parts[-1])

        img_path = os.path.join(BASE_DATASET_DIR, item.get('image', f"{item_id}/image.png"))
        mask_path = os.path.join(BASE_DATASET_DIR, item.get('bbox_mask', f"{item_id}/bbox_mask.png"))
        json_path = os.path.join(BASE_DATASET_DIR, f"{item_id}/view_features.json")

        rgb = cv2_imread_unicode(img_path, cv2.IMREAD_COLOR)
        bbox_mask = cv2_imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)

        if rgb is None or bbox_mask is None or not os.path.exists(json_path):
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        orig_H, orig_W = rgb.shape[:2]
        scale = SIZE / max(orig_H, orig_W)
        new_H, new_W = int(orig_H * scale), int(orig_W * scale)

        # --- Extract GT Target Feature Lines ---
        gt_lines = []
        for edge in json_data.get("edges", []):
            pts = edge.get("points", [])
            if not pts or len(pts) < 2:
                continue
            face_id = get_highest_priority_faceid(edge.get("faceRefs", []))
            if face_id == gt_faceid:
                pts_array = np.array(pts, np.float32) * scale
                pts_array = np.round(pts_array).astype(np.int32).reshape((-1, 1, 2))
                gt_lines.append(pts_array)

        if not gt_lines:
            continue

        # --- Image Preprocessing ---
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (new_W, new_H), interpolation=cv2.INTER_AREA)
        bbox_mask_resized = cv2.resize(bbox_mask, (new_W, new_H), interpolation=cv2.INTER_NEAREST)

        pad_h, pad_w = SIZE - new_H, SIZE - new_W
        rgb_padded = np.pad(rgb_resized, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        bbox_mask_padded = np.pad(bbox_mask_resized, ((0, pad_h), (0, pad_w)), mode='constant')

        rgb_norm = (np.transpose(rgb_padded.astype(np.float32) / 255.0, (2, 0, 1)) - MEAN) / STD
        bbox_mask_norm = np.expand_dims(bbox_mask_padded.astype(np.float32) / 255.0, axis=0)

        input_tensor = torch.from_numpy(np.concatenate([rgb_norm, bbox_mask_norm], axis=0)).float().unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(bbox_mask_norm).float().unsqueeze(0).to(device)

        # --- Model Inference ---
        with torch.no_grad():
            pred_offsets = model(input_tensor)

        dx = pred_offsets[0, 0, :, :].cpu().numpy()
        dy = pred_offsets[0, 1, :, :].cpu().numpy()

        y_grid, x_grid = np.mgrid[0:SIZE, 0:SIZE]
        x_new = x_grid + dx
        y_new = y_grid + dy

        valid_mask = bbox_mask_padded > 127
        if not np.any(valid_mask):
            continue

        # ==================================================
        # 📊 Metrics 1-4: Geometric Distance & Dispersion Statistics
        # ==================================================
        temp_edge_mask.fill(0)
        cv2.polylines(temp_edge_mask, gt_lines, isClosed=False, color=255, thickness=1)

        # 🌟 Core Fix 1: If GT lines are completely outside the image (cropped), skip distance statistics for this sample
        if not np.any(temp_edge_mask > 0):
            continue

        inv_gt_mask = cv2.bitwise_not(temp_edge_mask)
        dist_map = cv2.distanceTransform(inv_gt_mask, cv2.DIST_L2, 3)

        # Strictly restrict to predicted points inside the mask
        pred_x = np.clip(np.round(x_new[valid_mask]).astype(int), 0, SIZE - 1)
        pred_y = np.clip(np.round(y_new[valid_mask]).astype(int), 0, SIZE - 1)

        dists = dist_map[pred_y, pred_x]

        # 🌟 Core Fix 2: Double security, filtering out NaN, Inf, and garbage values > 2000 px
        valid_dists = dists[(np.isfinite(dists)) & (dists < 2000)]

        pts_count = len(valid_dists)
        if pts_count == 0:
            continue

        total_mask_pts += pts_count
        total_exact_hits += np.sum(valid_dists < 0.5)
        total_5px_hits += np.sum(valid_dists <= 5.0)
        total_10px_hits += np.sum(valid_dists <= 10.0)
        total_distance_sum += np.sum(valid_dists, dtype=np.float64)  # Force float64 to prevent overflow

        # ==================================================
        # 📉 Metric 5: Reconstruct GT Offsets & Compute Loss & Directional Consistency
        # ==================================================
        gt_pixels = np.argwhere(temp_edge_mask > 0)
        if len(gt_pixels) > 0:
            tree = cKDTree(gt_pixels)
            mask_pixels = np.argwhere(valid_mask)

            _, indices = tree.query(mask_pixels)
            closest_gt = gt_pixels[indices]

            gt_dy = closest_gt[:, 0] - mask_pixels[:, 0]
            gt_dx = closest_gt[:, 1] - mask_pixels[:, 1]

            # Calculate Directional Consistency (Cosine Similarity)
            pred_dx_val = dx[valid_mask]
            pred_dy_val = dy[valid_mask]

            dot_product = pred_dx_val * gt_dx + pred_dy_val * gt_dy
            norm_pred = np.sqrt(pred_dx_val ** 2 + pred_dy_val ** 2) + 1e-6
            norm_gt = np.sqrt(gt_dx ** 2 + gt_dy ** 2) + 1e-6
            cos_sim = dot_product / (norm_pred * norm_gt)

            valid_cos = cos_sim[np.isfinite(cos_sim)]
            total_cos_sim_sum += np.sum(valid_cos, dtype=np.float64)

            # Calculate Loss
            gt_offsets_np = np.zeros((2, SIZE, SIZE), dtype=np.float32)
            gt_offsets_np[0, valid_mask] = gt_dx
            gt_offsets_np[1, valid_mask] = gt_dy

            gt_offsets_tensor = torch.from_numpy(gt_offsets_np).float().unsqueeze(0).to(device)

            with torch.no_grad():
                loss = criterion(pred_offsets, gt_offsets_tensor, mask_tensor)
                total_loss += loss.item()
                valid_batches += 1

    # ==================================================
    # 🏆 Final Report Output
    # ==================================================
    if total_mask_pts == 0:
        print("⚠️ No valid test samples found!")
        return

    exact_ratio = (total_exact_hits / total_mask_pts) * 100
    ratio_5px = (total_5px_hits / total_mask_pts) * 100
    ratio_10px = (total_10px_hits / total_mask_pts) * 100

    mean_dispersion = total_distance_sum / total_mask_pts
    mean_cos_sim = total_cos_sim_sum / total_mask_pts
    avg_loss = total_loss / valid_batches if valid_batches > 0 else 0.0

    print("\n" + "=" * 60)
    print(" 🌟 Vector Field Prediction Quality Quantitative Report (Validation)")
    print("=" * 60)
    print(f" 🔹 Total Processed Mask Pixels : {total_mask_pts:,} pts")
    print("-" * 60)
    print(f" 🎯 Exact Hit Ratio (<0.5px)    : {exact_ratio:>6.2f}%  ({total_exact_hits:,} / {total_mask_pts:,})")
    print(f" 🎯 5px Tolerance Ratio (<=5px) : {ratio_5px:>6.2f}%  ({total_5px_hits:,} / {total_mask_pts:,})")
    print(f" 🎯 10px Tolerance Ratio(<=10px): {ratio_10px:>6.2f}%  ({total_10px_hits:,} / {total_mask_pts:,})")
    print("-" * 60)
    print(f" 📐 Mean Point Cloud Dispersion : {mean_dispersion:.4f} px")
    print(f" 🧭 Mean Directional Consistency: {mean_cos_sim:.4f} (Closer to 1 is better)")
    print(f" 📉 Total Validation Loss (Loss): {avg_loss:.6f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()











