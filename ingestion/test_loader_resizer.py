from Loader import Loader
from Resizer import Resizer

test_path = "./data/train/Andrena aerinifrons/"
fake_path = "./qjzeghdh"

# test config
noisy=True

# instanciate loader
loader = Loader()


# test on fake path
cv_img = loader.load_folder(fake_path, "jpg", noisy=noisy)
# test on real path
cv_img = loader.load_folder(test_path, "jpg", noisy=noisy)
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
cv_img_resized = resizer.resize(cv_img_list=cv_img, target_size=(512, 512), noisy=noisy)

# print result
print(f"Returned list of {len(cv_img_resized)} images")
try :
    print(f"shape of first image is {cv_img_resized[0].shape}")
except : pass

# save to new folder
if True :
    loader.save_img_to_folder(new_path="./IMAGE_SUPPRIME_RESIZED/", cv_img=cv_img_resized)


print("\n"*4)

# resize loaded images
cv_img_resized = resizer.expand_edges(cv_img_list=cv_img, target_size=(512, 512), noisy=noisy)

# print result
print(f"Returned list of {len(cv_img_resized)} images")
try :
    print(f"shape of first image is {cv_img_resized[0].shape}")
except : pass

# save to new folder
if True :
    loader.save_img_to_folder(new_path="./IMAGE_SUPPRIME_EXPANDED/", cv_img=cv_img_resized)
