import tensorflow as tf
import keras
from keras.losses import MeanSquaredError
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import sys
sys.path.append('.')
from utils.load_data import load_mnist
from model import DenoiseAutoencoder

def train():
    
    x_train, x_test = load_mnist()
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    
    # Add Noise
    noise_factor = 0.2
    x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
    x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

    x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
    x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)
    
    denoiser = DenoiseAutoencoder()
    
    print(denoiser.encoder.summary())
    
    denoiser.decoder.summary()
    
    denoiser.compile(optimizer='adam', loss=MeanSquaredError())
    
    os.makedirs("models", exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join("models", "denoiser_epoch_{epoch:02d}_valLoss_{val_loss:.4f}.keras"),
        monitor="val_loss",
        save_best_only=False,   # <-- save EVERY epoch
        save_weights_only=False,
        verbose=1
    )

    earlystop_cb = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    denoiser.fit(
        x_train_noisy, x_train,
        epochs=10,              # set higher since early stopping will cut it
        shuffle=True,
        validation_data=(x_test_noisy, x_test),
        callbacks=[checkpoint_cb, earlystop_cb]
    )
    
if __name__ == "__main__":
    train()