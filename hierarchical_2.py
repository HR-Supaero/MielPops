#######################################
# SECOND PASS — ONLY RARE CLASSES
#######################################

from ingestion.Loader import Loader
from ingestion.Resizer import Resizer
from ingestion.imbalance_equal import augmentation_1_species_all
import os
import pandas as pd
import cv2
import glob

OUT_SIZE = (224, 224)

image_path = "./data/train/"
second_level_path = "./data/hierarchical_level2/"

THRESHOLD_PARENT = 108   # définition de "rare"
THRESHOLD_CHILD = 13     # seuil d’augmentation

OTHERS_NAME = "others"

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
# Utils
#######################################

def save_images_unique(images, output_path, prefix):
    os.makedirs(output_path, exist_ok=True)
    existing = len(glob.glob(os.path.join(output_path, "*.jpg")))

    for i, img in enumerate(images):
        filename = f"{prefix}_{existing + i:06d}.jpg"
        cv2.imwrite(os.path.join(output_path, filename), img)

#######################################
# Main loop (ONLY < 108)
#######################################

for species in all_folders:

    n_images = image_counts[species]

    # ❌ on ignore les classes non rares
    if n_images >= THRESHOLD_PARENT:
        continue

    print("\n" + "="*40)
    print(f"LEVEL 2 — {species} ({n_images} images)")
    print("="*40)

    current_path = os.path.join(image_path, species)

    cv_img = loader.load_folder(current_path, "jpg", noisy=False)

    cv_img_resized = resizer.auto_rescale_expand(
        cv_img_list=cv_img,
        target_size=OUT_SIZE,
        noisy=False
    )

    # 🔹 AUGMENTATION si assez d’images
    if n_images > THRESHOLD_CHILD:

        print(f"Augmenting {species}")
        cv_img_final = augmentation_1_species_all(
            cv_img_resized,
            species_name=species
        )

        output_path = os.path.join(second_level_path, species)

        loader.save_img_to_folder(
            new_path=output_path + "/",
            cv_img=cv_img_final
        )

    # 🔹 SINON → others
    else:
        print(f"Moving {species} to others")

        output_path = os.path.join(second_level_path, OTHERS_NAME)

        save_images_unique(
            images=cv_img_resized,
            output_path=output_path,
            prefix=species
        )

#######################################
# CSV generation — LEVEL 2 (REINDEXED)
#######################################

df = pd.read_csv("./data/train.csv")

# extraire le nom d'espèce
df['species'] = df['id'].str.split('/').str[1]

# ne garder que les classes < THRESHOLD_PARENT
df = df[df['species'].apply(lambda x: image_counts[x] < THRESHOLD_PARENT)]

# regroupement : classes > THRESHOLD_CHILD gardées, sinon "others"
df['species'] = df['species'].apply(
    lambda x: x if image_counts[x] > THRESHOLD_CHILD else "others"
)

# label original pour others
df.loc[df['species'] == "others", 'label'] = 99

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
    "./data/hierarchical_level2.csv",
    index=False
)

# mapping hiérarchique → label original (pour l'inférence)
pd.DataFrame.from_dict(
    hier_to_label,
    orient="index",
    columns=["original_label"]
).to_csv("./data/hierarchical_label_mapping_level2.csv")

print("LEVEL 2 CSV + label mapping saved")