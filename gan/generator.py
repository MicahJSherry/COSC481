import tensorflow as tf
from tensorflow.keras import layers
def make_conv():
    model = tf.keras.Sequential()
    model.add(layers.Input((1000,)))

    
    model.add(layers.Dense(7*7*256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    return model

def make_generator_model():
    
    noise = tf.keras.Input(shape=(1000,))
    
    r = r_channel = make_conv()(noise)
    g = g_channel = make_conv()(noise)
    b = b_channel = make_conv()(noise)

    x = tf.keras.layers.Concatenate()([r,g,b])
    assert x.shape == (None, 28, 28, 3)
    
    model = tf.keras.Model(inputs=noise, outputs=x)

    return model



def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)
