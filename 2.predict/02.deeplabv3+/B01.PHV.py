import os
import sys
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("❌ Missing segmentation_models_pytorch library! Please run: pip install segmentation-models-pytorch")
    sys.exit(1)

# ==================================================
# 1. Configuration & Path Specifications
# ==================================================
# 💡 Path to the DeepLabV3+ weight file
MODEL_PATH = "best_vector_field_model_deeplabv3plus_1024.pth"
VAL_JSON_PATH = "../output_dataset_batch/val.json"
BASE_DATASET_DIR = "../output_dataset_batch"
OUTPUT_DIR = "faceid-output-deeplab-hardvoting"  # 💡 Dedicated output directory to distinguish experiments

SIZE = 1024
STRIDE = 16  # Sampling stride for regions outside the Mask
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🌟 Automatically create directories for correct and wrong predictions
os.makedirs(os.path.join(OUTPUT_DIR, "correct"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "wrong"), exist_ok=True)

# Priority mapping dictionary
PRIORITY_MAP = {"Highest": 4, "High": 3, "Medium": 2, "Low": 1}


# ==================================================
# 2. Helper Functions
# ==================================================
def cv2_imread_unicode(filepath, flags=cv2.IMREAD_COLOR):
    """Reads image files containing Unicode characters in paths."""
    if not os.path.exists(filepath):
        return None
    return cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), flags)


def cv2_imwrite_unicode(file_path, img):
    """Writes image files supporting Unicode characters in paths."""
    cv2.imencode('.png', img)[1].tofile(str(file_path))


def get_highest_priority_faceid(face_refs):
    """Extracts the FaceID with the highest priority from faceRefs."""
    if not face_refs:
        return 0
    sorted_refs = sorted(face_refs, key=lambda x: PRIORITY_MAP.get(x.get("priority", "Low"), 0), reverse=True)
    return sorted_refs[0]["faceID"]


# ==================================================
# 3. Main Validation & Visualization Pipeline
# ==================================================
def main():
    print("🚀 Loading DeepLabV3+ model...")

    # 💡 Instantiate DeepLabV3+ model (must align perfectly with train.py)
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,  # No pre-trained weights needed for validation; load local checkpoint directly
        in_channels=4,
        classes=2,
        activation=None
    )

    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device)
        # Smart state_dict key stripping (compatible with DataParallel weights)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            elif k.startswith('model.'):
                k = k[6:]
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        print("✅ DeepLabV3+ weights loaded successfully!")
    else:
        print(f"❌ Weight file not found: {MODEL_PATH}")
        return

    model.to(device)
    model.eval()

    with open(VAL_JSON_PATH, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    print(f"📦 Includes augmented samples (aug). Total validation samples: {len(val_data)}")

    correct_count = 0
    valid_count = 0

    print(f"\n🎯 Starting evaluation (DeepLabV3+ | Baseline: Hard Voting, With Aug)...")
    pbar = tqdm(val_data, desc="Validation Progress", unit="sample", colour="green")

    temp_edge_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)

    for item in pbar:
        item_id = item['id']

        # Safely extract gt_faceid, compatible with _aug suffix
        parts = item_id.split('_')
        if parts[-1] == 'aug':
            gt_faceid = int(parts[-2])
        else:
            gt_faceid = int(parts[-1])

        img_path = os.path.join(BASE_DATASET_DIR, item.get('image', f"{item_id}/image.png"))
        mask_path = os.path.join(BASE_DATASET_DIR, item.get('bbox_mask', f"{item_id}/bbox_mask.png"))
        json_path = os.path.join(BASE_DATASET_DIR, f"{item_id}/view_features.json")

        rgb = cv2_imread_unicode(img_path, cv2.IMREAD_COLOR)
        bbox_mask = cv2_imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)

        if rgb is None or bbox_mask is None or not os.path.exists(json_path):
            tqdm.write(f"⚠️ Skipping {item_id}: Missing required files")
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        edges = json_data.get("edges", [])

        orig_H, orig_W = rgb.shape[:2]
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # --- Image Preprocessing ---
        scale = SIZE / max(orig_H, orig_W)
        new_H, new_W = int(orig_H * scale), int(orig_W * scale)

        rgb_resized = cv2.resize(rgb, (new_W, new_H), interpolation=cv2.INTER_AREA)
        bbox_mask_resized = cv2.resize(bbox_mask, (new_W, new_H), interpolation=cv2.INTER_NEAREST)

        pad_h, pad_w = SIZE - new_H, SIZE - new_W
        rgb_padded = np.pad(rgb_resized, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        bbox_mask_padded = np.pad(bbox_mask_resized, ((0, pad_h), (0, pad_w)), mode='constant')

        # --- Model Inference ---
        rgb_norm = rgb_padded.astype(np.float32) / 255.0
        rgb_norm = np.transpose(rgb_norm, (2, 0, 1))
        rgb_norm = (rgb_norm - MEAN) / STD

        bbox_mask_norm = bbox_mask_padded.astype(np.float32) / 255.0
        bbox_mask_tensor = np.expand_dims(bbox_mask_norm, axis=0)

        input_tensor = np.concatenate([rgb_norm, bbox_mask_tensor], axis=0)
        input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)

        dx = output[0, 0, :, :].cpu().numpy()
        dy = output[0, 1, :, :].cpu().numpy()

        y_grid, x_grid = np.mgrid[0:SIZE, 0:SIZE]
        x_new = x_grid + dx
        y_new = y_grid + dy

        valid_mask = bbox_mask_padded > 127
        if not np.any(valid_mask):
            continue

        # Get coordinates of all predicted points
        pred_x = np.clip(np.round(x_new[valid_mask]).astype(int), 0, SIZE - 1)
        pred_y = np.clip(np.round(y_new[valid_mask]).astype(int), 0, SIZE - 1)

        # ==================================================
        # 🔎 Core: Baseline 1 - Pixel-Level Hard Voting
        # ==================================================
        faceid_votes = defaultdict(int)  # Stores pure vote counts
        faceid_lines = defaultdict(list)  # Records all line segments for visualization

        for edge in edges:
            pts = edge.get("points", [])
            if not pts or len(pts) < 2:
                continue

            face_id = get_highest_priority_faceid(edge.get("faceRefs", []))
            if face_id == 0:
                continue

            pts_array = np.array(pts, np.float32) * scale
            pts_array = np.round(pts_array).astype(np.int32).reshape((-1, 1, 2))
            faceid_lines[face_id].append(pts_array)

            # Clear temporary Mask
            temp_edge_mask.fill(0)

            # Draw current edge independently (line width of 3 pixels as hard tolerance boundary)
            cv2.polylines(temp_edge_mask, [pts_array], isClosed=False, color=1, thickness=3)

            # Check how many predicted points fall on this edge (boolean intersection)
            hits = temp_edge_mask[pred_y, pred_x]
            hit_count = np.sum(hits)

            if hit_count > 0:
                faceid_votes[face_id] += hit_count

        if faceid_votes:
            pred_faceid = max(faceid_votes, key=faceid_votes.get)
        else:
            pred_faceid = 0

        # --- Statistics & Accuracy ---
        valid_count += 1
        is_correct = (str(pred_faceid) == str(gt_faceid))
        if is_correct:
            correct_count += 1

        pbar.set_postfix({
            "Total": valid_count,
            "Correct": correct_count,
            "Accuracy": f"{(correct_count / valid_count) * 100:.1f}%"
        })

        # ==================================================
        # 🎨 OpenCV Fast Visualization & Concatenation
        # ==================================================
        # ---------------- Left Panel Processing ----------------
        img_left = rgb_padded.copy()

        # 1. Background Unification: Replace pure black/white backgrounds with light gray (240, 240, 240)
        h, w = img_left.shape[:2]
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]

        for pt in corners:
            if np.all(img_left[pt[1], pt[0]] <= 30):
                cv2.floodFill(img_left, flood_mask, pt, (240, 240, 240),
                              (30, 30, 30), (30, 30, 30), cv2.FLOODFILL_FIXED_RANGE)
            elif np.all(img_left[pt[1], pt[0]] >= 225):
                cv2.floodFill(img_left, flood_mask, pt, (240, 240, 240),
                              (30, 30, 30), (30, 30, 30), cv2.FLOODFILL_FIXED_RANGE)

        # 2. Draw Mask Covered Area (Semi-transparent Green)
        overlay = img_left.copy()
        overlay[valid_mask] = [0, 180, 0]
        img_left = cv2.addWeighted(overlay, 0.25, img_left, 0.75, 0)

        invalid_mask = ~valid_mask
        sample_mask = np.zeros_like(invalid_mask)
        sample_mask[::STRIDE, ::STRIDE] = True
        outside_mask = invalid_mask & sample_mask

        # 3. Plot Sparse Points Outside the Mask
        plot_x_out, plot_y_out = x_new[outside_mask], y_new[outside_mask]
        for px, py in zip(plot_x_out, plot_y_out):
            if 0 <= px < SIZE and 0 <= py < SIZE:
                cv2.circle(img_left, (int(px), int(py)), 3, (255, 190, 0), -1)

        # 4. Plot Dense Points Inside the Mask
        plot_x_in, plot_y_in = x_new[valid_mask], y_new[valid_mask]
        for px, py in zip(plot_x_in[::4], plot_y_in[::4]):
            if 0 <= px < SIZE and 0 <= py < SIZE:
                cv2.circle(img_left, (int(px), int(py)), 3, (255, 0, 0), -1)

        # ---------------- Right Panel Processing ----------------
        img_right = np.full((SIZE, SIZE, 3), 240, dtype=np.uint8)

        all_candidate_lines = []
        for fid, lines in faceid_lines.items():
            all_candidate_lines.extend(lines)

        # 1. Draw All Candidate Base Lines
        if all_candidate_lines:
            cv2.polylines(img_right, all_candidate_lines, isClosed=False, color=(120, 120, 120), thickness=2)

        # 2. Draw GT Target Line in Pure Black
        if gt_faceid in faceid_lines:
            cv2.polylines(img_right, faceid_lines[gt_faceid], isClosed=False, color=(0, 0, 0), thickness=4)

        # 3. Draw Model Predicted Line in Green (Correct) or Red (Incorrect)
        if pred_faceid > 0 and pred_faceid in faceid_lines:
            pred_color = (0, 180, 0) if is_correct else (220, 0, 0)
            cv2.polylines(img_right, faceid_lines[pred_faceid], isClosed=False, color=pred_color, thickness=3)

        # 4. Overlay Model's Actual Predicted Points in Red (Semi-transparent)
        overlay_right = img_right.copy()
        for px, py in zip(pred_x[::4], pred_y[::4]):
            cv2.circle(overlay_right, (int(px), int(py)), 2, (255, 0, 0), -1)

        img_right = cv2.addWeighted(overlay_right, 0.6, img_right, 0.4, 0)

        # 5. Write Status Text
        text_color = (0, 150, 0) if is_correct else (200, 0, 0)
        cv2.putText(img_right, f"Pred: {pred_faceid} | GT: {gt_faceid}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)

        # Concatenate and Save
        result_img = np.hstack((img_left, img_right))
        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

        folder_name = "correct" if is_correct else "wrong"
        output_filepath = os.path.join(OUTPUT_DIR, folder_name, f"{item_id}_result.png")
        cv2_imwrite_unicode(output_filepath, result_bgr)

    print(f"\n🎉 Evaluation completed!")
    print(f"📂 Results categorized and saved to: {OUTPUT_DIR}/correct/ and {OUTPUT_DIR}/wrong/")


if __name__ == "__main__":
    main()





































