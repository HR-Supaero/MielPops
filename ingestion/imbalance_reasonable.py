import pickle
import cv2
import plotly.express as px
import numpy as np

import imbalance_viz

def show_cv2_plotly(img):
    # Convert BGR → RGB (IMPORTANT)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Show with Plotly
    fig = px.imshow(img_rgb)
    fig.update_layout(coloraxis_showscale=False)
    fig.show()

def augmentation_toupie_bleyblade(list_data):
    list_data_augmented = []
    for i, img in enumerate(list_data):
        # # Show result
        show_cv2_plotly(img=img)

        # Rotate
        rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
        rotated_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_45 = rotate_img(img,45)
        
        # # Show result
        # show_cv2_plotly(img=rotated_90)
        # show_cv2_plotly(img=rotated_180)
        # show_cv2_plotly(img=rotated_270)
        show_cv2_plotly(img=rotated_45)

        list_data_augmented += [img, rotated_90, rotated_180, rotated_270]
    return list_data_augmented

def rotate_img(img, angle):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # compute new bounding dimensions
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # adjust rotation matrix
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Rotation + taille originale mais coupe les images
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def augmentation_toupie_bleyblade_var(list_data, n_aug):
    list_data_augmented = []
    for i, img in enumerate(list_data):
        img_list = [img]
        for a in range(n_aug) :
            alpha = a*360/n_aug
            img_list.append(rotate_img(img=img, angle=alpha))
        list_data_augmented += img_list
    return list_data_augmented


def augmentation_1_species(list_data, species_name):
    current_n = df_counts["n_images"][species_name]
    n_aug = df_counts["n_images_to_add"][species_name]//current_n
    
    return augmentation_toupie_bleyblade_var(list_data=list_data, n_aug=n_aug)

if __name__ == "__main__":
    df_counts = imbalance_viz.count_images_per_class(train_dir=imbalance_viz.TRAIN_DIR)
    n_max = df_counts["n_images"].max()
    df_counts["n_images_to_add"] = np.minimum(df_counts["n_images"] * 50, n_max) - df_counts["n_images"]

    # LOADING RESIZED
    folder_name = "Andrena aerinifrons" # also folder name ?
    with open('data/Andrena_aerinifrons.pkl', 'rb') as f:
        list_data = pickle.load(f)
    print(len(list_data))

    list_data_aug = augmentation_1_species(list_data=list_data, species_name=folder_name)
    print(len(list_data_aug))

