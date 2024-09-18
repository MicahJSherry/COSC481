import os 
import cv2



def load_images_(dir):
    images = []
    labels = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path) 

                images.append(image)
                labels.append(root)
    
    return image, labels