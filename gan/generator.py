import tensorflow as tf
from tensorflow.keras import layers
def make_conv(in_dim):
    model = tf.keras.Sequential()
    model.add(layers.Input((in_dim,)))

    
    model.add(layers.Dense(7*7*256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    return model



def make_large_conv(in_dim):
    model = tf.keras.Sequential()
    model.add(layers.Input((in_dim,)))

    model.add(layers.Dense(16*16*256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((16, 16, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    # Assert the new output shape
    assert model.output_shape == (None, 128, 128, 1)
    return model 



def make_generator_model(in_dim=100, outshape= (28,28, 1)):
    
    noise = tf.keras.Input(shape=(in_dim,))
    assert outshape[0]==outshape[1]
    
    if outshape[0] ==128:
        conv = make_large_conv
    else:      
        conv = make_conv
    
    if outshape[-1]== 3: 
        k = conv(in_dim)(noise)*.50 
        r = conv(in_dim)(noise)
        g = conv(in_dim)(noise)
        b = conv(in_dim)(noise)
        x = tf.keras.layers.Concatenate()([k+r,k+g,k+b])
    
    else: 
        x = conv(in_dim)(noise) 
         
    model = tf.keras.Model(inputs=noise, outputs=x)

    return model


def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)
