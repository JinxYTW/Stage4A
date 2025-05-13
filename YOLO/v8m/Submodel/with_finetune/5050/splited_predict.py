import os
import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
# === PARAMÈTRES ===
MODEL_PATH = "my_model.pt"  # Peut être modifié
INPUT_DIR = "img"           # Dossier d'entrée
OUTPUT_DIR = "predictions"  # Dossier de sortie
TILE_SIZE = 320             # Taille des tuiles (modifiable)
CONF_THRESHOLD = 0.05  # À adapter selon les besoins
# ====================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Charger le modèle YOLO
model = YOLO(MODEL_PATH)

# Fonction pour ajuster les coordonnées des prédictions selon le décalage
def adjust_boxes(boxes, offset_x, offset_y):
    # Cloner les données brutes (x1, y1, x2, y2, conf, cls)
    new_data = boxes.data.clone()
    new_data[:, [0, 2]] += offset_x  # x1, x2
    new_data[:, [1, 3]] += offset_y  # y1, y2

    # Créer un nouveau Boxes avec les mêmes dimensions originales
    adjusted = Boxes(new_data, boxes.orig_shape)
    return adjusted



# Boucle sur toutes les images du dossier
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

            # Filtrer par confiance
            
            filtered_boxes = [b for b in results.boxes if float(b.conf[0]) >= CONF_THRESHOLD]

            # Si des boxes restent, on les ajuste
            if filtered_boxes:
                box_tensors = torch.stack([b.data[0] for b in filtered_boxes])
                adjusted = Boxes(box_tensors, results.boxes.orig_shape)
                adjusted_boxes = adjust_boxes(adjusted, x, y)
                all_boxes.extend(adjusted_boxes)


    # Dessiner les détections
    for box in all_boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{model.names[cls_id]} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Compter les olives détectées (supposons que la classe "olive" a l’ID 0)
        olive_count = sum(int(box.cls[0]) == 0 for box in all_boxes)

        # Afficher le nombre en haut à gauche de l’image
        cv2.putText(image, f"Olives detectees : {olive_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    out_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(out_path, image)
    print(f"✅ Sauvegardé : {out_path}")

     # === Sauvegarde YOLO format texte ===
    label_dir = os.path.join(OUTPUT_DIR, "labels")
    os.makedirs(label_dir, exist_ok=True)
    label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")

    with open(label_path, "w") as f:
        for box in all_boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0]
            x_center = (x1 + x2) / 2 / W
            y_center = (y1 + y2) / 2 / H
            width = (x2 - x1) / W
            height = (y2 - y1) / H
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("✅ Traitement terminé !")
