from sklearn.cluster import KMeans
import cv2


image_path = "spark22/test/cheops/cheops_img03816.jpg"


image = cv2.imread(image_path)
original_shape = image.shape
image = image.reshape(-1, image.shape[-1])
print(original_shape)
print(image.shape)

kmeans = KMeans(n_clusters=2, n_init=100)
kmeans.fit(image)

image = kmeans.cluster_centers_[kmeans.labels_]
image = image.reshape(original_shape)
print(image)
cv2.imwrite("color.png",image.astype("uint8") )


