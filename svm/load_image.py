import os 
import cv2



def load_images(dir):
    images = []
    labels = []
    for root, dirs, files in os.walk(dir):
        for file in files[0:100]:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path, mode=cv2.IMREAD_GRAYSCALE)
                print(type(image))
                images.append(image)
                labels.append(root.split("/")[-1])
    
    return image, labels

print(load_images("./spark22"))