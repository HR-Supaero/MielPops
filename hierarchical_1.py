from ingestion.Loader import Loader
from ingestion.Resizer import Resizer
from ingestion.imbalance_reasonable import augmentation_1_species
from ingestion.imbalance_equal import augmentation_1_species_all
import os
import pandas as pd

OUT_SIZE = (224, 224)
image_path = "./data/train/"
hierarchical_path = "./data/hierarchical_level1/"
test_image_path = "./data/test"
treated_test_image_path = "./data/treated_test"



# get all class folders
all_files_and_folders = os.listdir(image_path)
all_folders = [f for f in all_files_and_folders if os.path.isdir(os.path.join(image_path, f))]
print(f"Found folders {all_folders}")
print("\n"*3)

# instanciate loader
loader = Loader()

# instanciate resizer 
resizer = Resizer()

print("instanciated loader and resizer")
print("\n"*3)


from collections import Counter
import os
import pandas as pd

image_counts = {}

for species in all_folders:
    species_path = os.path.join(image_path, species)
    image_counts[species] = len([f for f in os.listdir(species_path) if f.endswith(".jpg")])

THRESHOLD = 108
OTHERS_NAME = "others"

import cv2
import glob

def save_images_unique(images, output_path, prefix):
    os.makedirs(output_path, exist_ok=True)

    existing = len(glob.glob(os.path.join(output_path, "*.jpg")))

    for i, img in enumerate(images):
        filename = f"{prefix}_{existing + i:06d}.jpg"
        cv2.imwrite(os.path.join(output_path, filename), img)

for species in all_folders:

    current_path = image_path + species + "/"
    n_images = image_counts[species]

    print("\n" + "="*40)
    print(f"WORKING ON SPECIES {species} ({n_images} images)")
    print("="*40)

    print(f"Loading images of folder {species}...")
    cv_img = loader.load_folder(current_path, "jpg", noisy=False)
    print(f"... {len(cv_img)} images loaded")

    # resize (pour tout le monde)
    cv_img_resized = resizer.auto_rescale_expand(
        cv_img_list=cv_img,
        target_size=OUT_SIZE,
        noisy=False
    )

    # dossier de sortie
    if n_images > THRESHOLD:
        # AUGMENTATION
        cv_img_final = augmentation_1_species_all(
            cv_img_resized,
            species_name=species
        )
        output_species = species

        loader.save_img_to_folder(
            new_path=hierarchical_path + output_species + "/",
            cv_img=cv_img_final
        )

    else:
        # REGROUPEMENT DANS OTHERS
        output_species = OTHERS_NAME
        output_path = os.path.join(hierarchical_path, output_species)

        save_images_unique(
            images=cv_img_resized,
            output_path=output_path,
            prefix=species
        )


# ================================
# CSV CREATION — LEVEL 1 (REINDEXED)
# ================================

df = pd.read_csv("./data/train.csv")

# extraire le nom d'espèce
df['species'] = df['id'].str.split('/').str[1]

# regrouper les espèces rares dans "others"
df['species'] = df['species'].apply(
    lambda x: x if image_counts[x] > THRESHOLD else "others"
)

# label original
df.loc[df['species'] == "others", 'label'] = 99

# ne garder qu’une ligne par classe
df = df[['species', 'label']].drop_duplicates().reset_index(drop=True)

# 🔑 REINDEXATION DES LABELS
original_labels = sorted(df['label'].unique())
label_to_hier = {lbl: i for i, lbl in enumerate(original_labels)}
hier_to_label = {i: lbl for lbl, i in label_to_hier.items()}

df['hier_label'] = df['label'].map(label_to_hier)

# sauvegardes
df[['species', 'hier_label']].to_csv("./data/hierarchical_level1.csv", index=False)

# sauvegarde du mapping pour l'inférence
pd.DataFrame.from_dict(
    hier_to_label, orient="index", columns=["original_label"]
).to_csv("./data/hierarchical_label_mapping_level1.csv")

print("LEVEL 1 CSV + label mapping saved")
#######################################
# edit test images
#######################################

# print(f"Loading images of test folder...")
# cv_img = loader.load_folder(test_image_path, "jpg", noisy=False, keep_names=True)
# print(f"... {len(cv_img)} images of test folder loaded !")
# try :
#     print(f"Shape of first image is {cv_img[0].shape}")
# except : pass
# print("\n"*3)

# file_names = loader.get_loaded_file_names()

# print(file_names)

# # resize loaded images
# print(f"Resizing images of test folder...")
# cv_img_resized = resizer.auto_rescale_expand(cv_img_list=cv_img, target_size=OUT_SIZE, noisy=False)
# print(f"... {len(cv_img_resized)} images of test folder resized !")
# try :
#     print(f"Shape of first image is {cv_img_resized[0].shape}")
# except : pass
# print("\n"*3)

# if not(os.path.exists(treated_test_image_path)):
#         os.makedirs(treated_test_image_path)
# loader.save_img_to_folder(new_path = treated_test_image_path, 
#                                 cv_img=cv_img_resized, name_list=file_names)