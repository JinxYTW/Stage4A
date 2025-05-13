import os

# === Chemins ===
IMAGES_DIR = "images"
LABELS_DIR = "obj_train_data"
TRAIN_FILE = "train.txt"

# === Images présentes ===
existing_images = set(os.listdir(IMAGES_DIR))
existing_image_basenames = set(os.path.splitext(f)[0] for f in existing_images)

# === Nettoyage de train.txt ===
filtered_lines = []
deleted_labels = 0
train_image_basenames = set()

with open(TRAIN_FILE, "r") as f:
    lines = f.readlines()

for line in lines:
    image_path = line.strip()
    image_name = os.path.basename(image_path)
    image_basename, _ = os.path.splitext(image_name)
    label_file = os.path.join(LABELS_DIR, image_basename + ".txt")

    if image_name in existing_images:
        filtered_lines.append(line)
        train_image_basenames.add(image_basename)
    else:
        if os.path.exists(label_file):
            os.remove(label_file)
            deleted_labels += 1
        print(f"❌ Supprimé ligne train.txt + {label_file}")

# === Réécriture propre de train.txt ===
with open(TRAIN_FILE, "w") as f:
    f.writelines(filtered_lines)

print(f"✅ {len(lines) - len(filtered_lines)} lignes supprimées dans train.txt")
print(f"🗑️  {deleted_labels} fichiers labels supprimés (liés aux lignes supprimées)")

# === Suppression des fichiers .txt orphelins non listés dans train.txt ===
all_label_files = [f for f in os.listdir(LABELS_DIR) if f.endswith(".txt")]
orphans_deleted = 0

for label_file in all_label_files:
    label_basename = os.path.splitext(label_file)[0]
    if label_basename not in existing_image_basenames:
        label_path = os.path.join(LABELS_DIR, label_file)
        os.remove(label_path)
        orphans_deleted += 1
        print(f"🗑️  Supprimé label orphelin : {label_file}")

print(f"✅ {orphans_deleted} fichiers .txt orphelins supprimés dans {LABELS_DIR}")
