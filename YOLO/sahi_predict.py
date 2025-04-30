from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import torch
import cv2
import os

# === PARAMÈTRES ===
MODEL_PATH = "my_model.pt"  # Chemin vers votre modèle
INPUT_DIR = "img"           # Dossier contenant les images d'entrée
OUTPUT_DIR = "predictions_sahi"  # Dossier de sortie pour les résultats
TILE_SIZE = 320             # Taille des tuiles
OVERLAP_RATIO = 0.2         # Chevauchement entre les tuiles
CONF_THRESHOLD = 0.5       # Seuil de confiance

# Créer les dossiers de sortie si nécessaire
os.makedirs(OUTPUT_DIR, exist_ok=True)
label_dir = os.path.join(OUTPUT_DIR, "labels")
os.makedirs(label_dir, exist_ok=True)

# Déterminer le périphérique à utiliser
device = "cuda" if torch.cuda.is_available() else "cpu"

# Charger le modèle avec SAHI
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=MODEL_PATH,
    confidence_threshold=CONF_THRESHOLD,
    device=device,
)

# Boucle sur toutes les images du dossier
for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(INPUT_DIR, filename)
    image = cv2.imread(img_path)
    H, W = image.shape[:2]

    # Prédiction avec slicing SAHI
    result = get_sliced_prediction(
        image=image,
        detection_model=detection_model,
        slice_height=TILE_SIZE,
        slice_width=TILE_SIZE,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO,
        perform_standard_pred=False,
    )

    # Dessiner les prédictions sur l'image
    for obj in result.object_prediction_list:
        bbox = obj.bbox.to_xyxy()
        cls_id = int(obj.category.id)
        conf = obj.score.value

        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{obj.category.name} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Compter les objets détectés (par exemple, classe ID=0 pour "olive")
    olive_count = sum(int(obj.category.id) == 0 for obj in result.object_prediction_list)
    cv2.putText(image, f"Olives détectées : {olive_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Sauvegarder l'image annotée
    out_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(out_path, image)
    print(f"✅ Sauvegardé : {out_path}")

    # Sauvegarde des labels (format YOLO)
    label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
    with open(label_path, "w") as f:
        for obj in result.object_prediction_list:
            bbox = obj.bbox.to_xywh()
            cls_id = int(obj.category.id)
            x_center = (bbox[0] + bbox[2] / 2) / W
            y_center = (bbox[1] + bbox[3] / 2) / H
            width = bbox[2] / W
            height = bbox[3] / H
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("✅ Traitement terminé !")
