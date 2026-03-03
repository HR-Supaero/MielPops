import glob
import cv2
import os


"""
A class to load images using cv2.
Use as follow :

loader = Loader()
cv_img = loader.load_folder(path="./path", file_type="jpg", noisy=True)


"""
class Loader():
    def __init__(self):
        return None

    def convert_folder_to_jpg(self, path, file_types=["png", "jpeg"]):
        if path[-1] != "/":
            path += "/"
        if not os.path.isdir(path):
            print(f"Loader (convert) :: ERROR : PATH {path} DOES NOT EXIST")
            return None
        
        for file_type in file_types :
            pattern = f"{path}*.{file_type}"
            if noisy : print(f"Converting image like {pattern}")
            globpath = glob.glob(pattern)
            for img in globpath:
                im = Image.open(img)
                rgb_im = im.convert('RGB')
                rgb_im.save(img[:-len(file_type)] + ".jpg")
                print(f"Saved image {img[:-len(file_type)]}.jpg to jpg")

    def load_folder(self, path, file_type="jpg", noisy=True, keep_names=False):
        if path[-1] != "/":
            path += "/"
        if not os.path.isdir(path):
            print(f"Loader (load) :: ERROR : PATH {path} DOES NOT EXIST")
            return None
        pattern = f"{path}*.{file_type}"
        if noisy : print(f"Loading image like {pattern}")
        globpath = glob.glob(pattern)
        if noisy : print("glob glob")
        cv_img = []
        self.names = []
        count_img = 0
        for img in globpath:
            readimg = cv2.imread(img)
            cv_img.append(readimg)
            count_img += 1 
            if noisy : print(f"Loaded image {count_img} : {img}")
            if keep_names : self.names.append(img.split("/")[-1]) # keep only the name of the file, not the directory
        print("Done !")
        return cv_img
    
    def get_loaded_file_names(self):
        try:
            return self.names
        except :
            print("self.names does not exist, execute load_folder with keep_names=True beforehand")
    
    def save_img_to_folder(self, new_path, cv_img, noisy=True, name_list=None):
        if new_path[-1] != "/":
            new_path += "/"
        # Create the directory
        try:
            os.mkdir(new_path)
            print(f"Directory '{new_path}' created successfully.")
        except FileExistsError:
            print(f"Directory '{new_path}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{new_path}'.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        # done with folder, writing
        count_img = 0
        for img in cv_img :
            # rename file if needed
            if name_list is None :
                export_file = f"{new_path}image_{count_img}.jpg"
            else :
                export_file = f"{new_path}{name_list[count_img]}"
            count_img += 1
            cv2.imwrite(export_file, img)
            if noisy : print(f"Written image {count_img} to {export_file}")
        return None



    def get_train_folder_from_parent(self, parent_path, file_type="jpg", noisy=True):
        return None
    

         
