from ingestion.Loader import Loader
from ingestion.Resizer import Resizer
from ingestion.imbalance_reasonable import augmentation_1_species


#######################################
# RESIZING
#######################################


test_path = "./data/train/Andrena aerinifrons/"
species = "Andrena aerinifrons"

# instanciate loader
loader = Loader()

# test on real path
print(f"Loading images of folder {species}...")
cv_img = loader.load_folder(test_path, "jpg", noisy=True)
print(f"... {len(cv_img)} images of folder {species} loaded !")
try :
    print(f"Shape of first image is {cv_img[0].shape}")
except : pass
print("\n"*3)

# instanciate resizer 
resizer = Resizer()

# resize loaded images
print(f"Resizing images of folder {species}...")
cv_img_resized = resizer.resize(cv_img_list=cv_img, size=(512, 512), noisy=True)
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
if False :
    loader.save_img_to_folder(new_path="./RESIZED_IMAGE_SUPPRIME/", cv_img=cv_img_resized)
