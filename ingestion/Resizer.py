import cv2
import os


"""
A class to resize loaded images using cv2.
Use as follow :


"""
class Resizer():
    def __init__(self):
        return None 
    
    def resize(self, cv_img_list, size=(512, 512), noisy=True):
        resized_img_list = []
        count_img = 0 
        for img in cv_img_list :
            old_shape = img.shape
            new_img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_AREA)
            resized_img_list.append(new_img)
            count_img += 1 
            if noisy : print(f"Resized image {count_img} from {old_shape} to {size}")
        return resized_img_list
