from sklearn.cluster import KMeans
import cv2
import os
import numpy as np

image_paths = dir_list = os.listdir("images")
print(image_paths)

for image_path in image_paths:

    image = cv2.imread("images/"+image_path)
    original_shape = image.shape
    
    
    image = image.reshape(-1, image.shape[-1])

    kmeans = KMeans(n_clusters=5, n_init=10)
    kmeans.fit(image)
    centers = kmeans.cluster_centers_
    
    image = centers[kmeans.labels_]
    image = image.reshape(original_shape)
    cv2.imwrite("segmented_images/"+image_path,image.astype("uint8") )


