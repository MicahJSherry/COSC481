import tensorflow as tf
import matplotlib.pyplot as plt

from generator import make_generator_model

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
print(generated_image)
plt.imsave('noise.png', generated_image, cmap='viridis') 
