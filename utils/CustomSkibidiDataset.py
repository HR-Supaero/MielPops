from torch.utils.data import Dataset
from torchvision import datasets
import os
import pandas as pd
from torchvision.io import decode_image



class CustomSkibidiDataset(Dataset):
    def __init__(self, img_dir, csv_mapper, transform=None, target_transform=None, noisy=True):
        # go to the data dir and list the folders there. Each folder corresponds to a class.
        all_files_and_folders = os.listdir(img_dir)
        all_folders = [f for f in all_files_and_folders if os.path.isdir(os.path.join(img_dir, f))]
        if noisy : print(f"Found folders {all_folders}")
        
        # load the species to id dataframe
        df_id_mapper = pd.read_csv(csv_mapper)
        id_to_label = dict(zip(df_id_mapper['species'], df_id_mapper['label']))
        print(id_to_label)

        # Create the DataFrame
        rows = []
        for folder_class in all_folders :
            if noisy : print(f"Working on class folder {folder_class}")
            current_folder_dir = img_dir + folder_class + "/"
            files_in_class = os.listdir(current_folder_dir)
            for file in files_in_class:
                rows.append((current_folder_dir + file, id_to_label[folder_class]))
            if noisy : print("-- done")
        
        # setting things up
        self.img_labels = pd.DataFrame(rows, columns=["file", "class"])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def get_df(self):
        return self.img_labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img_path = self.img_labels.iloc[idx, 0]
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label