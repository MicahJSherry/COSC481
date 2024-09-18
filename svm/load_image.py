import os 
import cv2
import numpy as np


def load_images(dir,limit=100):
    images = []
    labels = []
    for root, dirs, files in os.walk(dir):
        for file in files[0:limit]:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                print(image.shape)
                image = cv2.resize(image, None, fx=0.5, fy=0.5)

                images.append(np.reshape(image,(-1,)))
                
                labels.append(root.split("/")[-1])
    
    return np.vstack(images), labels

