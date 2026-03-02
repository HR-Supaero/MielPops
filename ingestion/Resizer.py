import cv2
import os
import numpy as np


"""
A class to resize loaded images using cv2.
Use as follow :

loader = Loader()
cv_img = loader.load_folder(path="./path", file_type="jpg", noisy=True)
resizer = Resizer()
cv_img_resized = resizer.resize(cv_img_list=cv_img, target_size=(512, 512), noisy=True)

"""
class Resizer():
    def __init__(self):
        return None 
    
    def resize(self, cv_img_list, target_size=(512, 512), noisy=True):
        resized_img_list = []
        count_img = 0 
        for img in cv_img_list :
            old_shape = img.shape
            new_img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_AREA)
            resized_img_list.append(new_img)
            count_img += 1 
            if noisy : print(f"Resized image {count_img} from {old_shape} to {target_size}")
        return resized_img_list
    
    def expand_edges(self, cv_img_list, target_size=(512, 512), noisy=True, blur=False):
        expanded_img_list = []
        count_img = 0 
        for img in cv_img_list :
            old_shape = img.shape
            if old_shape == target_size :
                if noisy : print(f"image already has target shape of {target_size}")
            # expand the edges of the image
            new_img = self._expand_edges_one_image(img, target_size=target_size, 
                                                    border_type=cv2.BORDER_REPLICATE, value=0, blur=blur)
            expanded_img_list.append(new_img)
            count_img += 1 
            if noisy : print(f"Expanded image {count_img} edges from {old_shape} to {target_size}")
        return expanded_img_list
    
    def auto_rescale_expand(self, cv_img_list, target_size=(512, 512), noisy=True, blur=False):
        # rescale image if it is small, expand it if it is big and mix the two when needed
        expanded_img_list = []
        count_img = 0 
        for img in cv_img_list :
            old_shape = img.shape
            if old_shape == target_size :
                if noisy : print(f"image already has target shape of {target_size}")
                new_image = np.copy(img)
            else :
                # image needs resizing
                factor_w = target_size[0] / old_shape[0]
                factor_h = target_size[1] / old_shape[1]
                # how much the image can be distorted and still fit in
                global_extension_factor = min(factor_h, factor_w)
                # resize
                resized_img = cv2.resize(img, None, fx=global_extension_factor, 
                                            fy=global_extension_factor, interpolation=cv2.INTER_AREA)

                # expand the edges of the image
                new_img = self._expand_edges_one_image(resized_img, target_size=target_size, 
                                                    border_type=cv2.BORDER_REPLICATE, value=0, blur=blur)
            expanded_img_list.append(new_img)
            count_img += 1 
            if noisy : print(f"Expanded image {count_img} edges from {old_shape} to {target_size}")
        return expanded_img_list



    def _expand_edges_one_image(self, img, target_size=(512, 512), border_type=cv2.BORDER_REPLICATE, value=0, blur=False):
        h, w = img.shape[:2]
        target_w, target_h = target_size

        # Calculate padding for width and height
        pad_w = max(0, target_w - w)
        pad_h = max(0, target_h - h)

        # Split padding equally on both sides (left/right, top/bottom)
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        # Apply padding
        expanded_img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                border_type, value=value)
        
        if blur :
            # blurring size
            blur_kernel_size = (min(2*(pad_h//4)+1, 2*(target_h//40)+1), 
                                min(2*(pad_w//4)+1, 2*(target_w//40)+1))
            # blur whole image
            blurred_expanded_img = cv2.GaussianBlur(expanded_img, blur_kernel_size, 0)
            # mask the edges
            mask_edges = np.zeros((target_h, target_w, 3))
            bottom = None if bottom==0 else -bottom
            right = None if right==0 else -right
            mask_edges[top:bottom, left:right] = 1
            # keep the blur when it is not an edge
            out = np.where(mask_edges==1, expanded_img, blurred_expanded_img)
            # replace the image
            expanded_img = out
        return expanded_img
    



