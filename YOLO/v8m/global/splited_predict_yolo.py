import os
import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
import zipfile
import shutil

# === PARAMÈTRES ===
MODEL_PATH = "my_model.pt"
INPUT_DIR = "img"
OUTPUT_DIR = "dataset_yolo"
TILE_SIZE = 320
CONF_THRESHOLD = 0.05

# === DOSSIERS ===
FLAT_DIR = os.path.join(OUTPUT_DIR, "obj_train_data")  # images + labels ici
NAMES_FILE = os.path.join(OUTPUT_DIR, "obj.names")
DATA_FILE = os.path.join(OUTPUT_DIR, "obj.data")
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.txt")
ZIP_FILE = "yolo_dataset.zip"

os.makedirs(FLAT_DIR, exist_ok=True)

# === Charger le modèle ===
model = YOLO(MODEL_PATH)

def adjust_boxes(boxes, offset_x, offset_y):
    new_data = boxes.data.clone()
    new_data[:, [0, 2]] += offset_x
    new_data[:, [1, 3]] += offset_y
    return Boxes(new_data, boxes.orig_shape)

train_image_paths = []

# === Traitement image par image ===
for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(INPUT_DIR, filename)
    image = cv2.imread(img_path)
    H, W = image.shape[:2]
    all_boxes = []

    for y in range(0, H, TILE_SIZE):
        for x in range(0, W, TILE_SIZE):
            tile = image[y:y+TILE_SIZE, x:x+TILE_SIZE]
            if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                continue

            results = model(tile, verbose=False)[0]
            filtered_boxes = [b for b in results.boxes if float(b.conf[0]) >= CONF_THRESHOLD]

            if filtered_boxes:
                box_tensors = torch.stack([b.data[0] for b in filtered_boxes])
                adjusted = Boxes(box_tensors, results.boxes.orig_shape)
                adjusted_boxes = adjust_boxes(adjusted, x, y)
                all_boxes.extend(adjusted_boxes)

    # === Sauvegarde image ===
    flat_image_path = os.path.join(FLAT_DIR, filename)
    cv2.imwrite(flat_image_path, image)
    train_image_paths.append(f"obj_train_data/{filename}")

    # === Sauvegarde label (même nom que l'image) ===
    label_path = os.path.join(FLAT_DIR, os.path.splitext(filename)[0] + ".txt")
    with open(label_path, "w") as f:
        if all_boxes:
            for box in all_boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0]
                x_center = (x1 + x2) / 2 / W
                y_center = (y1 + y2) / 2 / H
                width = (x2 - x1) / W
                height = (y2 - y1) / H
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        # sinon fichier vide (obligatoire)

# === obj.names ===
with open(NAMES_FILE, "w") as f:
    for i in range(len(model.names)):
        f.write(f"{model.names[i]}\n")

# === obj.data ===
with open(DATA_FILE, "w") as f:
    f.write(f"classes = {len(model.names)}\n")
    f.write(f"train = train.txt\n")
    f.write(f"names = obj.names\n")
    f.write("backup = backup/\n")

# === train.txt ===
with open(TRAIN_FILE, "w") as f:
    for rel_path in train_image_paths:
        f.write(f"{rel_path}\n")

# === ZIP complet du dataset ===
if os.path.exists(ZIP_FILE):
    os.remove(ZIP_FILE)

with zipfile.ZipFile(ZIP_FILE, "w") as zipf:
    for folder, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            full_path = os.path.join(folder, file)
            rel_path = os.path.relpath(full_path, OUTPUT_DIR)
            zipf.write(full_path, arcname=rel_path)

print(f"✅ Dataset YOLO 1.1 prêt pour CVAT : {ZIP_FILE}")

