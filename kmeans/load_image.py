import os 
import cv2
import numpy as np

from imgbeddings import imgbeddings
ibed = imgbeddings()

def load_images(dir,limit=100):
    model = cv2.dnn.readNetFromCaffe('model.prototxt', 'model.caffemodel')
    
    images = []
    labels = []
    for root, dirs, files in os.walk(dir):
        for file in files[0:limit]:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                image = cv2.resize(image,(224, 224))

                embedding = ibed.to_embeddings(image)
                print(embedding.shape)
                images.append(images)
                
                labels.append(root.split("/")[-1])
    
    return np.vstack(images), labels

