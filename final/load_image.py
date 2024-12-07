import os 
import cv2
import numpy as np
import kagglehub


def load_images(dir,limit=100, Shape=(224,224)):
    images = []
    labels = []
    for root, dirs, files in os.walk(dir):
        for file in files[0:limit]:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                
                image = cv2.resize(image, Shape)

                images.append(image) 
                labels.append(root.split("/")[-1])
    
    return np.array(images), labels



def download_sat_images():

    path = kagglehub.dataset_download("sunray2333/whurs191")
    print(f"Path to dataset files: {path}")
    
    return path

