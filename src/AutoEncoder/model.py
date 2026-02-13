import keras
import tensorflow as tf
from keras import Sequential
from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose
from keras.models import Model

class DenoiseAutoencoder(Model):
  def __init__(self):
    super(DenoiseAutoencoder, self).__init__()
    self.encoder = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
        Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)
    ])
    self.decoder = Sequential([
        Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
        Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
        Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')  # to match the input channel size
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded