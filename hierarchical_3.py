#######################################
# THIRD PASS — VERY RARE CLASSES (NO OTHERS)
#######################################

from ingestion.Loader import Loader
from ingestion.Resizer import Resizer
import os
import pandas as pd

OUT_SIZE = (224, 224)

image_path = "./data/train/"
level3_path = "./data/hierarchical_level3/"

THRESHOLD_CHILD = 13

#######################################
# Setup
#######################################

loader = Loader()
resizer = Resizer()

all_folders = [
    f for f in os.listdir(image_path)
    if os.path.isdir(os.path.join(image_path, f))
]

#######################################
# Count images
#######################################

image_counts = {
    species: len([
        f for f in os.listdir(os.path.join(image_path, species))
        if f.endswith(".jpg")
    ])
    for species in all_folders
}

#######################################
# Main loop — ONLY < 13 IMAGES
#######################################

for species in all_folders:

    n_images = image_counts[species]

    # ❌ on ignore tout sauf les classes très rares
    if n_images >= THRESHOLD_CHILD:
        continue

    print("\n" + "="*40)
    print(f"LEVEL 3 — {species} ({n_images} images)")
    print("="*40)

    current_path = os.path.join(image_path, species)

    cv_img = loader.load_folder(current_path, "jpg", noisy=False)

    cv_img_resized = resizer.auto_rescale_expand(
        cv_img_list=cv_img,
        target_size=OUT_SIZE,
        noisy=False
    )

    output_path = os.path.join(level3_path, species)
    os.makedirs(output_path, exist_ok=True)

    loader.save_img_to_folder(
        new_path=output_path + "/",
        cv_img=cv_img_resized
    )

#######################################
# CSV generation — LEVEL 3 (REINDEXED)
#######################################

df = pd.read_csv("./data/train.csv")

# extraire le nom d'espèce
df['species'] = df['id'].str.split('/').str[1]

# ne garder que les classes < THRESHOLD_CHILD
df = df[df['species'].apply(lambda x: image_counts[x] < THRESHOLD_CHILD)]

# une ligne par classe
df = df[['species', 'label']].drop_duplicates().reset_index(drop=True)

# =========================
# 🔑 REINDEXATION DES LABELS
# =========================

original_labels = sorted(df['label'].unique())

label_to_hier = {lbl: i for i, lbl in enumerate(original_labels)}
hier_to_label = {i: lbl for lbl, i in label_to_hier.items()}

df['hier_label'] = df['label'].map(label_to_hier)

# =========================
# SAUVEGARDES
# =========================

# CSV utilisé pour l'entraînement (labels continus)
df[['species', 'hier_label']].to_csv(
    "./data/hierarchical_level3.csv",
    index=False
)

# mapping hiérarchique → label original
pd.DataFrame.from_dict(
    hier_to_label,
    orient="index",
    columns=["original_label"]
).to_csv("./data/hierarchical_label_mapping_level3.csv")

print("LEVEL 3 CSV + label mapping saved")