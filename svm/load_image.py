import os 
import cv2
import np


def load_images(dir,limit=100):
    images = []
    labels = []
    for root, dirs, files in os.walk(dir):
        for file in files[0:limit]:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image = np.reshape(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE),(-1,))
                
                images.append(image)
                labels.append(root.split("/")[-1])
    
    return images, labels

print(load_images("./spark22"))