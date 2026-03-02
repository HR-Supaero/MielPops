import pickle
import cv2
import plotly.express as px
import numpy as np
import pandas as pd
import random
from pathlib import Path

def show_cv2_plotly(img):
    # Convert BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig = px.imshow(img_rgb)
    fig.update_layout(coloraxis_showscale=False)
    fig.show()

TRAIN_DIR = "data/train"

def count_images_per_class(train_dir: str, extension: str = ".jpg") -> pd.DataFrame:
    """
    Parcourt un dossier train et compte le nombre d'images par sous-dossier.

    Parameters
    ----------
    train_dir : str
        Chemin vers data/train
    extension : str
        Extension des fichiers image à compter

    Returns
    -------
    pd.DataFrame
        DataFrame indexée par nom de classe avec colonne 'n_images'
    """

    train_path = Path(train_dir)

    data = []

    # Parcours des sous-dossiers (espèces)
    for subfolder in sorted(train_path.iterdir()):
        if subfolder.is_dir():
            n_images = len(list(subfolder.glob(f"*{extension}")))

            data.append({
                "class_name": subfolder.name,
                "n_images": n_images
            })

    # Création dataframe
    df = pd.DataFrame(data)
    df = df.set_index("class_name").sort_values("n_images", ascending=False)

    return df

# ----------------------------
# Rotation aléatoire
# ----------------------------
def rotate_img_random(img, max_angle=360):
    """
    Rotation aléatoire entre -max_angle/2 et +max_angle/2
    """
    angle = random.uniform(-max_angle/2, max_angle/2)
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Bounding dimensions
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust rotation matrix
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Apply rotation
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# ----------------------------
# Flips aléatoires
# ----------------------------
def random_flip(img):
    choice = random.choice([0, 1, -1, None])  # horizontal, vertical, both, or none
    if choice is not None:
        img = cv2.flip(img, choice)
    return img

# ----------------------------
# Augmentation complète aléatoire
# ----------------------------
import random

def augment_list_random(list_data, n_aug=5, show=False, show_prob=0.1):
    """
    list_data : liste d'images (numpy arrays)
    n_aug : nombre d'images aléatoires à générer par image
    show : afficher certaines transformations avec plotly
    show_prob : probabilité d'afficher une image transformée (0-1)
    """
    list_data_aug = []

    for img in list_data:
        # Ajouter l'image originale
        list_data_aug.append(img)
        if show and random.random() < show_prob:
            show_cv2_plotly(img)

        for _ in range(n_aug):
            img_aug = img.copy()
            # Rotation aléatoire
            img_aug = rotate_img_random(img_aug, max_angle=360)
            # Flip aléatoire
            img_aug = random_flip(img_aug)

            list_data_aug.append(img_aug)
            # Affichage aléatoire selon show_prob
            if show and random.random() < show_prob:
                show_cv2_plotly(img_aug)

    return list_data_aug


def augmentation_1_species_all(list_data, species_name):
    df_counts = count_images_per_class(train_dir=TRAIN_DIR)
    n_max = df_counts["n_images"].max()
    df_counts["n_images_to_add"] = n_max - df_counts["n_images"]
    
    current_n = df_counts["n_images"][species_name]
    n_aug = df_counts["n_images_to_add"][species_name]//current_n
    
    return augment_list_random(list_data=list_data, n_aug=n_aug, show=True)

if __name__ == "__main__":
    # LOADING RESIZED
    folder_name = "Andrena aerinifrons" # also folder name ?
    with open('data/Andrena_aerinifrons.pkl', 'rb') as f:
        list_data = pickle.load(f)
    print(len(list_data))

    list_data_aug = augmentation_1_species_all(list_data=list_data, species_name=folder_name)
    print(len(list_data_aug))