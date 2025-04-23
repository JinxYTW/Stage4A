import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import DeformableDetrForObjectDetection, AutoImageProcessor
from torchvision.ops import box_convert, nms

# === PARAMÈTRES ===
MODEL_DIR = "output/final_model"
PROCESSOR_DIR = "output/final_processor"
INPUT_DIR = "img"
OUTPUT_DIR = "predictions"
LABEL_DIR = "predictions/labels"
TILE_SIZE = 320
CONF_THRESHOLD = 0.8
CLASS_TO_COUNT = 0
# ===================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeformableDetrForObjectDetection.from_pretrained(MODEL_DIR).to(device)
processor = AutoImageProcessor.from_pretrained(PROCESSOR_DIR)

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(INPUT_DIR, filename)
    image = cv2.imread(img_path)
    H, W = image.shape[:2]

    full_img_annotated = image.copy()
    all_boxes = []
    all_scores = []
    all_labels = []

    for y in range(0, H - TILE_SIZE + 1, TILE_SIZE):
        for x in range(0, W - TILE_SIZE + 1, TILE_SIZE):
            tile = image[y:y+TILE_SIZE, x:x+TILE_SIZE]
            pil_tile = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
            inputs = processor(images=pil_tile, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits.softmax(-1)[0]
            keep = logits.max(-1).values > CONF_THRESHOLD
            if keep.sum() == 0:
                continue

            boxes = outputs.pred_boxes[0][keep].cpu()
            scores = logits[keep].max(-1).values.cpu()
            labels = logits[keep].argmax(-1).cpu()

            boxes *= TILE_SIZE
            boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
            boxes = boxes.clamp(min=0, max=TILE_SIZE)

            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box.tolist()

                # Translation : faire du coin bas-droit le nouveau centre
                cx = x2
                cy = y2
                dx = cx - (x1 + x2) // 2
                dy = cy - (y1 + y2) // 2

                x1_new = int(x1 + dx)
                y1_new = int(y1 + dy)
                x2_new = int(x2 + dx)
                y2_new = int(y2 + dy)

                # Clamp dans les limites de la tuile
                x1_new = max(0, min(TILE_SIZE, x1_new))
                y1_new = max(0, min(TILE_SIZE, y1_new))
                x2_new = max(0, min(TILE_SIZE, x2_new))
                y2_new = max(0, min(TILE_SIZE, y2_new))

                # Reprojection dans l'image globale
                x1_full = x1_new + x
                y1_full = y1_new + y
                x2_full = x2_new + x
                y2_full = y2_new + y

                all_boxes.append([x1_full, y1_full, x2_full, y2_full])
                all_scores.append(score.item())
                all_labels.append(label.item())

    if len(all_boxes) == 0:
        print(f"❌ {filename} : aucune détection.")
        continue

    # === Appliquer NMS global ===
    all_boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
    all_scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
    all_labels_tensor = torch.tensor(all_labels, dtype=torch.int64)

    keep = nms(all_boxes_tensor, all_scores_tensor, iou_threshold=0.5)
    kept_boxes = all_boxes_tensor[keep]
    kept_scores = all_scores_tensor[keep]
    kept_labels = all_labels_tensor[keep]

    total_count = 0
    yolo_lines = []

    for box, label, score in zip(kept_boxes, kept_labels, kept_scores):
        x1, y1, x2, y2 = map(int, box.tolist())
        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        w = (x2 - x1) / W
        h = (y2 - y1) / H

        yolo_lines.append(f"{label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        cv2.rectangle(full_img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(full_img_annotated, f"C{label} {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        if label.item() == CLASS_TO_COUNT:
            total_count += 1

    # Sauvegarde des labels YOLO
    label_filename = os.path.splitext(filename)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_filename)
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))

    # Ajout du compteur total
    cv2.putText(full_img_annotated, f"Total olives (class {CLASS_TO_COUNT}) : {total_count}",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_final.jpg")
    cv2.imwrite(out_path, full_img_annotated)
    print(f"✅ {filename} traité : {total_count} objets détectés, labels → {label_path}")

print("🎯 Tous les fichiers ont été traités avec succès.")
