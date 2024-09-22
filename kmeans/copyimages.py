import os 
import cv2
import numpy as np


def copy_image_dir_subset(dir,num_files_per_dir, out_dir = "images"):
    for root, dirs, files in os.walk(dir):
        for file in files[0:num_files_per_dir]:
            if file.endswith('.jpg'):
                origin_path = os.path.join(root, file)
                image = cv2.imread(origin_path)
            
                cv2.imwrite(out_dir+"/"+file, image)

copy_image_dir_subset("spark22/test", 2, out_dir = "images")                
