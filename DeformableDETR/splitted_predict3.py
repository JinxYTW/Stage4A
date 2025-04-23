import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import DeformableDetrForObjectDetection, AutoImageProcessor
from torchvision.ops import box_convert

# === PARAMÈTRES ===
MODEL_DIR = "output/final_model"
PROCESSOR_DIR = "output/final_processor"
INPUT_DIR = "img2"
OUTPUT_DIR = "predictions"
TILE_SIZE = 320
CONF_THRESHOLD = 0.8
CLASS_TO_COUNT = 0
# ===================

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeformableDetrForObjectDetection.from_pretrained(MODEL_DIR).to(device)
processor = AutoImageProcessor.from_pretrained(PROCESSOR_DIR)

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(INPUT_DIR, filename)
    image = cv2.imread(img_path)
    H, W = image.shape[:2]

    total_count = 0
    full_img_annotated = image.copy()

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

                # Reprojection dans l'image globale
                x1_full = int(x1 + x)
                y1_full = int(y1 + y)
                x2_full = int(x2 + x)
                y2_full = int(y2 + y)

                # Dessin sur l'image globale
                cv2.rectangle(full_img_annotated, (x1_full, y1_full), (x2_full, y2_full), (0, 255, 0), 2)
                cv2.putText(full_img_annotated, f"C{label.item()} {score:.2f}", 
                            (x1_full, y1_full - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                if label.item() == CLASS_TO_COUNT:
                    total_count += 1

    # Ajouter le compteur total
    cv2.putText(full_img_annotated, f"Total olives (class {CLASS_TO_COUNT}) : {total_count}",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_final.jpg")
    cv2.imwrite(out_path, full_img_annotated)
    print(f"✅ {filename} traité, {total_count} olives détectées → sauvegardé sous : {out_path}")

print("🎯 Tous les fichiers ont été traités avec succès.")
