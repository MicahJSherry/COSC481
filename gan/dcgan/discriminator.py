import tensorflow as tf 
from tensorflow.keras import layers

def make_1_channel(inshape):
    model = tf.keras.Sequential()
   
    model.add(layers.Input(inshape))

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    return model


def make_discriminator_model(inshape):
    image = tf.keras.Input(inshape) 
    if inshape[-1]==3:
        r = make_1_channel(inshape)(image[:,:,:,0:1])
        g = make_1_channel(inshape)(image[:,:,:,1:2])
        b = make_1_channel(inshape)(image[:,:,:,2:3])
        
        x = tf.keras.layers.Concatenate()([r,g,b])
        x = layers.Dense(15)(x)
        x = layers.Dense(10)(x)
    else:
        x = make_1_channel(inshape)(image)
    x = layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=image, outputs=x)
    return model

def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
