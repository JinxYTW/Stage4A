import os
import cv2
import torch
from PIL import Image
from transformers import DeformableDetrForObjectDetection, AutoImageProcessor
from torchvision.ops import box_convert

# === PARAMÈTRES ===
MODEL_DIR = "output/final_model"
PROCESSOR_DIR = "output/final_processor"
INPUT_DIR = "img2"
OUTPUT_DIR = "tiles_output"
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

    tile_id = 0
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

           

            # === Convertir les boîtes de coordonnées normalisées en pixels ===
            boxes *= TILE_SIZE  # Multiplier par la taille de la tuile

             # Convertir les boîtes de cxcywh à xyxy
            boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

            # S'assurer que les boîtes sont à l'intérieur des limites de la tuile
            boxes = boxes.clamp(min=0, max=TILE_SIZE)

            # Vérification de la conversion des coordonnées
            #print(f"Boxes for tile {tile_id}: {boxes}")

            # === Dessin local sur la tuile ===
            class_counter = 0
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = map(int, box.tolist())

                # S'assurer que les coordonnées sont correctes et éviter les erreurs sur les bords
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(TILE_SIZE, x2), min(TILE_SIZE, y2)

                cv2.rectangle(tile, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(tile, f"C{label.item()} {score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(tile, (cx, cy), 3, (0, 0, 255), -1)

                if label.item() == CLASS_TO_COUNT:
                    class_counter += 1

            # Ajouter un compteur de classe pour affichage
            cv2.putText(tile, f"Count {class_counter}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            out_tile_path = os.path.join(
                OUTPUT_DIR,
                f"{os.path.splitext(filename)[0]}_tile_{tile_id:03}.jpg"
            )
            cv2.imwrite(out_tile_path, tile)
            tile_id += 1

print("🔍 Tuiles traitées individuellement. Résultats dans tiles_output/")
