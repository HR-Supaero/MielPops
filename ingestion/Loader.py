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

    def load_folder(self, path, file_type="jpg", noisy=True):
        if path[-1] != "/":
            path += "/"
        if not os.path.isdir(path):
            print(f"Loader :: ERROR : PATH {path} DOES NOT EXIST")
            return None
        pattern = f"{path}*.{file_type}"
        if noisy : print(f"Loading image like {pattern}")
        globpath = glob.glob(pattern)
        if noisy : print("glob glob")
        cv_img = []
        count_img = 0
        for img in globpath:
            readimg = cv2.imread(img)
            cv_img.append(readimg)
            count_img += 1 
            if noisy : print(f"Loaded image {count_img} : {img }")
        print("Done !")
        return cv_img
    
    def save_img_to_folder(self, new_path, cv_img, noisy=True):
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
            count_img += 1
            export_file = f"{new_path}image_{count_img}.jpg"
            cv2.imwrite(export_file, img)
            if noisy : print(f"Written image {count_img} to {export_file}")
        return None



    def get_train_folder_from_parent(self, parent_path, file_type="jpg", noisy=True):
        return None
    

         
