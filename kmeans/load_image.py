import os 
import numpy as np
from PIL import Image

from imgbeddings import imgbeddings

def load_images(dir,limit=100):
    labels = []
    paths = []
    for root, dirs, files in os.walk(dir):
        for file in files[0:limit]:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                paths.append(image_path)
                labels.append(root.split("/")[-1])
    
    ibed = imgbeddings()
    emb = ibed.to_embeddings(paths)
    return np.vstack(emb), labels

