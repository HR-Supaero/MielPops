from Loader import Loader
from Resizer import Resizer

test_path = "./data/train/Andrena aerinifrons/"
fake_path = "./qjzeghdh"

# instanciate loader
loader = Loader()


# test on fake path
cv_img = loader.load_folder(fake_path, "jpg", noisy=True)
# test on real path
cv_img = loader.load_folder(test_path, "jpg", noisy=True)
#

# print result
print(f"Returned list of {len(cv_img)} images")
try :
    print(f"shape of first image is {cv_img[0].shape}")
except : pass

print("\n"*10)

# instanciate resizer 
resizer = Resizer()

# resize loaded images
cv_img_resized = resizer.resize(cv_img_list=cv_img, size=(512, 512), noisy=True)

# print result
print(f"Returned list of {len(cv_img_resized)} images")
try :
    print(f"shape of first image is {cv_img_resized[0].shape}")
except : pass

# save to new folder
if True :
    loader.save_img_to_folder(new_path="./RESIZED_IMAGE_SUPPRIME/", cv_img=cv_img_resized)
