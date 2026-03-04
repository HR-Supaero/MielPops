"""
Run this script to augment the data and save it in a new folder. It will also resize the images to 224*224.
"""

from ingestion.Loader import Loader
from ingestion.Resizer import Resizer
from ingestion.imbalance_reasonable import augmentation_1_species
import os

OUT_SIZE = (224, 224)
image_path = "./data/train/"
treated_image_path = "./data/treated_train/"
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


#######################################
# edit train images
#######################################
for species in all_folders :
    current_path = image_path + species + "/"
    print("\n\n" + "="*30 )
    print(f"WORKING ON SPECIES {species}")
    print(f"in folder {current_path}")
    print("="*30 +"\n\n")
    
    #######################################
    # RESIZING
    #######################################

    # test on real path
    print(f"Loading images of folder {species}...")
    cv_img = loader.load_folder(current_path, "jpg", noisy=False)
    print(f"... {len(cv_img)} images of folder {species} loaded !")
    try :
        print(f"Shape of first image is {cv_img[0].shape}")
    except : pass
    print("\n"*3)



    # resize loaded images
    print(f"Resizing images of folder {species}...")
    cv_img_resized = resizer.auto_rescale_expand(cv_img_list=cv_img, target_size=OUT_SIZE, noisy=False)
    print(f"... {len(cv_img_resized)} images of folder {species} resized !")
    try :
        print(f"Shape of first image is {cv_img_resized[0].shape}")
    except : pass
    print("\n"*3)

    #######################################
    # DATA AUGMENTATION
    #######################################

    # Less imbalanced output but still not equal classes
    print(f"Augmenting number of images of folder {species}...")
    cv_img_augmented = augmentation_1_species(cv_img_resized, species_name=species)
    print(f"... {len(cv_img_resized)} images of folder {species} have been transformed in {len(cv_img_augmented)} images !")
    print("\n"*3)




    # save to new folder
    if not(os.path.exists(treated_image_path + species + "/")):
        os.makedirs(treated_image_path + species + "/")
    loader.save_img_to_folder(new_path = treated_image_path + species + "/", 
                                cv_img=cv_img_augmented)

#######################################
# edit test images
#######################################

print(f"Loading images of test folder...")
cv_img = loader.load_folder(test_image_path, "jpg", noisy=False, keep_names=True)
print(f"... {len(cv_img)} images of test folder loaded !")
try :
    print(f"Shape of first image is {cv_img[0].shape}")
except : pass
print("\n"*3)

file_names = loader.get_loaded_file_names()

print(file_names)

# resize loaded images
print(f"Resizing images of test folder...")
cv_img_resized = resizer.auto_rescale_expand(cv_img_list=cv_img, target_size=OUT_SIZE, noisy=False)
print(f"... {len(cv_img_resized)} images of test folder resized !")
try :
    print(f"Shape of first image is {cv_img_resized[0].shape}")
except : pass
print("\n"*3)

if not(os.path.exists(treated_test_image_path)):
        os.makedirs(treated_test_image_path)
loader.save_img_to_folder(new_path = treated_test_image_path, 
                                cv_img=cv_img_resized, name_list=file_names)