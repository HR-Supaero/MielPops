import pickle
import cv2
import plotly.express as px
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
        # show_cv2_plotly(img=img)

        # Rotate
        rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
        rotated_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # # Show result
        # show_cv2_plotly(img=rotated_90)
        # show_cv2_plotly(img=rotated_180)
        # show_cv2_plotly(img=rotated_270)

        list_data_augmented += [img, rotated_90, rotated_180, rotated_270]
    return list_data_augmented


if __name__ == "__main__":
    df_counts = imbalance_viz.count_images_per_class(train_dir=imbalance_viz.TRAIN_DIR)
    print(df_counts)
    n_max = df_counts.max().val
    print(n_max)


    # # LOADING RESIZED
    # with open('data/Andrena_aerinifrons.pkl', 'rb') as f:
    #     list_data = pickle.load(f)

    # print(len(list_data))

    # list_data_augmented = augmentation_toupie_bleyblade(list_data=list_data)
    # print(len(list_data_augmented))

